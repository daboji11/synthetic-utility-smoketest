#!/usr/bin/env python
"""Fine-tune official pretrained native DP on exactly 28 H50 left human demos."""
import argparse
import datetime
import json
from pathlib import Path
import random
import sys
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_open_drawer_pretrained_dp import DEFAULT_CHECKPOINT
from src.workflows.native_dp_finetune import H50PretrainedFineTuneWorkspace,normalizer_equal

DATASET=ROOT/"data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256"
OUTPUT=ROOT/"outputs/OpenDrawer/dp_pretrained_human_left/seed_0"

def build_config(checkpoint,dataset,output,steps=10000,batch_size=4,learning_rate=1e-5):
    OmegaConf.register_new_resolver("now",lambda fmt:datetime.datetime.now().strftime(fmt),replace=True)
    cfg=OmegaConf.load(checkpoint.parent.parent/".hydra/config.yaml")
    if steps<1 or (steps>500 and steps%500):
        raise ValueError("Use <=500 steps or a multiple of500")
    if batch_size<1 or learning_rate<=0:
        raise ValueError("Invalid batch size/learning rate")
    task=OmegaConf.load(ROOT/"configs/native_dp/task/h50_open_drawer_left.yaml")
    task.dataset._target_="src.workflows.native_dp_finetune.PretrainedNormalizerH50Dataset"
    task.dataset.dataset_paths=[str(dataset.resolve())]
    normalizer=checkpoint.parent.parent/"normalizer.pkl"
    task.dataset.pretrained_normalizer_path=str(normalizer.absolute())
    task.ckpt_path=None
    cfg.task=task
    cfg._target_="src.workflows.native_dp_finetune.H50PretrainedFineTuneWorkspace"
    cfg.task_name="h50_open_drawer_left"
    cfg.exp_name="pretrained_ema_h50_left_finetune"
    cfg.dataloader.batch_size=batch_size
    cfg.dataloader.num_workers=4
    cfg.dataloader.persistent_workers=False
    cfg.val_dataloader.batch_size=batch_size
    cfg.val_dataloader.num_workers=0
    cfg.optimizer.learning_rate=learning_rate
    cfg.training.device="cuda:0"
    cfg.training.seed=0
    cfg.training.debug=False
    cfg.training.resume=False
    cfg.training.use_ema=True
    cfg.training.gradient_accumulate_every=1
    cfg.training.num_epochs=max(1,steps//500)
    cfg.training.max_train_steps=min(500,steps)
    cfg.training.lr_warmup_steps=min(500,max(0,steps//10))
    cfg.training.checkpoint_every=1
    cfg.training.rollout_every=None
    cfg.training.sample_every=1000000
    cfg.checkpoint.topk.k=0
    cfg.checkpoint.save_last_ckpt=True
    cfg.checkpoint.save_last_snapshot=False
    cfg.logging.mode="disabled"
    cfg.logging.name="h50_left_native_dp_seed_0"
    cfg.logging.project="h50_left_native_dp_finetune"
    cfg.logging.resume=False
    cfg.native_finetune={
        "source_checkpoint":str(checkpoint.absolute()),
        "source_revision":"c484448aba1a9b60a04c9b0ca117241518ea69f3",
        "pretrained_normalizer_path":str(normalizer.absolute()),
        "steps":steps,"save_freq":2500,"language_text":"open the left drawer",
        "episodes":28,"fps":20,"output":str(output.absolute()),
        "architecture_unchanged":True,"synthetic_data":False,
        "initialization":"published EMA into both models; fresh optimizer and counters"}
    return OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))

def preflight(workspace):
    dataset=hydra.utils.instantiate(workspace.cfg.task.dataset)
    if not normalizer_equal(dataset.get_normalizer(),workspace.model.normalizer):
        raise ValueError("Dataset adapter changed checkpoint normalizer")
    if any(not torch.equal(value,workspace.ema_model.state_dict()[key])
           for key,value in workspace.model.state_dict().items()):
        raise ValueError("Train model/EMA initialization differs")
    sample=dataset.datasets[0][0]
    batch_size=workspace.cfg.dataloader.batch_size
    batch={"obs":{key:value.unsqueeze(0).repeat(batch_size,*([1]*value.ndim)).to("cuda")
                  for key,value in sample["obs"].items()},
           "action":sample["action"].unsqueeze(0).repeat(batch_size,1,1).to("cuda")}
    workspace.model.to("cuda").eval()
    shapes={}
    hooks=[]
    for camera in ["robot0_agentview_right_image","robot0_agentview_left_image","robot0_eye_in_hand_image"]:
        hooks.append(workspace.model.obs_encoder.obs_nets[camera].register_forward_pre_hook(
            lambda module,inputs,key=camera:shapes.update({key:list(inputs[0].shape)})))
    try:
        with torch.inference_mode():
            loss=float(workspace.model.compute_loss(batch))
    finally:
        for hook in hooks:
            hook.remove()
    if not np.isfinite(loss) or len(shapes)!=3 or workspace.optimizer.state:
        raise ValueError("Invalid preflight forward/optimizer/cameras")
    result={"passed":True,"episodes":28,"frames":len(dataset),"fps":20,"batch_size":batch_size,
            "initialization":"EMA -> model+EMA; optimizer state empty; counters0",
            "normalizer_equal_pretrained":True,"camera_encoder_inputs":shapes,
            "forward_loss":loss,"optimizer_updates":0,"backward_executed":False,
            "gpu_peak_allocated_mb":torch.cuda.max_memory_allocated()/1024**2,
            "output":workspace.output_dir}
    print("FINETUNE_PREFLIGHT_OK "+json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",type=Path,default=DATASET)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.output.exists() and not args.preflight:
        raise FileExistsError(f"Preserving existing run: {args.output}")
    torch.set_num_threads(4)
    cfg=build_config(args.checkpoint,args.dataset,args.output,args.steps,args.batch_size,args.learning_rate)
    workspace=H50PretrainedFineTuneWorkspace(cfg,output_dir=str(args.output.absolute()))
    if args.preflight:
        preflight(workspace)
        return
    print(f"FINETUNE_START episodes=28 fps=20 batch_size={args.batch_size} steps={args.steps} "
          f"lr={args.learning_rate} seed=0 output={args.output}",flush=True)
    workspace.run()
if __name__=="__main__":
    main()
