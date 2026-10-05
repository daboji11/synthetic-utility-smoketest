#!/usr/bin/env python
"""Official LeRobot Diffusion trainer with native H50 guards and exact per-step metrics."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import draccus
import numpy as np
import torch
from lerobot.configs import FeatureType
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.feature_utils import dataset_to_policy_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.workflows.open_drawer_diffusion import (
    BASE_OUTPUT, CAMERAS, DATASET, REPO_ID, REQUIRED_MODEL_FILES,
    policy_protocol, validate_dataset,
)
CONFIG = ROOT/"configs/experiment/open_drawer_h50_human_left_3cam_diffusion.yaml"

def load_config(args):
    cfg = draccus.parse(TrainPipelineConfig, config_path=str(CONFIG), args=[])
    cfg.dataset.root = DATASET
    cfg.batch_size = args.batch_size or (1 if args.smoke else cfg.batch_size)
    cfg.steps = args.steps or (200 if args.smoke else cfg.steps)
    cfg.save_freq = cfg.steps if args.smoke else 2500
    cfg.log_freq = 10
    name = args.run_name or (f"smoke_seed_0_batch_{cfg.batch_size}" if args.smoke else "seed_0")
    if not name.isascii() or not name.replace("_","").isalnum():
        raise ValueError("Run name must be ASCII letters, digits, underscores")
    if cfg.batch_size < 1 or cfg.steps < 1 or (args.smoke and not 200 <= cfg.steps <= 500):
        raise ValueError("Smoke requires 200..500 steps; positive batch size")
    cfg.output_dir = BASE_OUTPUT/name
    if args.smoke and name == "seed_0":
        raise ValueError("Smoke must not occupy the formal seed_0 output")
    cfg.validate()
    if cfg.policy.type != "diffusion" or cfg.policy.pretrained_path or cfg.resume:
        raise ValueError("Fresh official Diffusion only; no ACT or pretrained robot policy")
    # Compare every architecture, horizon, scheduler and normalization field to the installed defaults.
    defaults = DiffusionConfig(device="cuda")
    allowed = {"input_features","output_features","device","push_to_hub","use_amp"}
    for key,value in draccus.encode(defaults).items():
        if key not in allowed and draccus.encode(cfg.policy).get(key) != value:
            raise ValueError(f"Non-default Diffusion parameter prohibited: {key}")
    return cfg

def preflight(cfg):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support required")
    dataset = validate_dataset()
    features = dataset_to_policy_features(dataset.features)
    cfg.policy.input_features = {k:v for k,v in features.items() if v.type != FeatureType.ACTION}
    cfg.policy.output_features = {k:v for k,v in features.items() if v.type == FeatureType.ACTION}
    cfg.policy.validate_features()
    sample=dataset[0]
    if list(cfg.policy.image_features) != CAMERAS:
        raise ValueError("All three cameras required in fixed order")
    print("DATASET_SCHEMA "+json.dumps({
        "episodes":dataset.num_episodes,"frames":len(dataset),"fps":dataset.meta.fps,
        "instruction":"open the left drawer","image_keys":CAMERAS,
        "image_shapes":{k:list(sample[k].shape) for k in CAMERAS},
        "state_shape":list(sample["observation.state"].shape),
        "action_shape":list(sample["action"].shape),"no_resampling":True,
    }),flush=True)
    print("DIFFUSION_CONFIG "+json.dumps({
        **policy_protocol(cfg.policy),"batch_size":cfg.batch_size,
        "steps":cfg.steps,"seed":cfg.seed,"output_dir":str(cfg.output_dir),
        "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "diffusers":importlib.metadata.version("diffusers"),
        "optimizer":draccus.encode(cfg.optimizer),"lr_scheduler":draccus.encode(cfg.scheduler),
    }),flush=True)

def train_worker(cfg):
    from lerobot.scripts import lerobot_train as official
    # Allowed mixed precision only; model, camera count, horizons and training implementation unchanged.
    torch.set_autocast_dtype("cuda",torch.bfloat16)
    torch.set_num_threads(4)
    started=time.perf_counter()
    first_update_time=None
    rows=[]
    camera_audit={"forward":{},"backward":{}}
    handles=[]
    original_make_policy=official.make_policy
    original_update=official.update_policy
    original_save=official.save_checkpoint
    metrics_file=None

    def checked_make_policy(*args,**kwargs):
        policy=original_make_policy(*args,**kwargs)
        cfg.output_dir.mkdir(parents=True,exist_ok=True)
        (cfg.output_dir/"resolved_config.json").write_text(json.dumps(cfg.to_dict(),indent=2,default=str)+"\n")
        shutil.copy2(CONFIG,cfg.output_dir/"requested_config.yaml")
        print(f"POLICY_CREATED official Diffusion parameters={sum(p.numel() for p in policy.parameters())}",flush=True)
        if len(policy.diffusion.rgb_encoder) != 3:
            raise ValueError("Expected three independent official RGB encoders")
        for key,encoder in zip(CAMERAS,policy.diffusion.rgb_encoder,strict=True):
            def forward_hook(module,inputs,output, key=key):
                if key in camera_audit["forward"]:
                    return
                images=inputs[0]
                expected=current_batch[key].flatten(0,1)
                if images.shape != expected.shape or not torch.equal(images,expected):
                    raise ValueError(f"Encoder did not receive the matching camera: {key}")
                camera_audit["forward"][key]={
                    "input_shape":list(images.shape),"output_shape":list(output.shape),
                    "matches_preprocessed_training_camera":True}
            def gradient_hook(grad,key=key):
                if key not in camera_audit["backward"]:
                    norm=float(grad.float().norm().item())
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError(f"Invalid visual gradient: {key}={norm}")
                    camera_audit["backward"][key]={"first_weight_grad_norm":norm,"finite_nonzero":True}
                return grad
            handles.append(encoder.register_forward_hook(forward_hook))
            handles.append(next(encoder.parameters()).register_hook(gradient_hook))
        original_forward=policy.forward
        def finite_forward(*args,**kwargs):
            output=original_forward(*args,**kwargs)
            if not torch.isfinite(output[0]).all():
                raise ValueError("Non-finite Diffusion loss: stopped before backward")
            return output
        # The official updater invokes .forward directly, so a module forward hook would be skipped.
        policy.forward=finite_forward
        return policy

    current_batch={}
    def measured_update(*args,**kwargs):
        nonlocal first_update_time, metrics_file,current_batch
        tracker,policy,batch=args[:3]
        current_batch=batch
        if tuple(batch["observation.state"].shape) != (cfg.batch_size,2,16):
            raise ValueError("Observation history/state shape mismatch")
        if tuple(batch["action"].shape) != (cfg.batch_size,64,12):
            raise ValueError("Diffusion prediction-target shape mismatch")
        if any(tuple(batch[key].shape) != (cfg.batch_size,2,3,128,128) for key in CAMERAS):
            raise ValueError("Temporal camera shapes mismatch")
        if first_update_time is None:
            first_update_time=time.perf_counter()
        result=original_update(*args,**kwargs)
        torch.cuda.synchronize()
        tracker=result[0]
        loss=float(tracker.loss.val)
        lr=float(tracker.lr.val)
        grad_norm=float(tracker.grad_norm.val)
        if not all(map(math.isfinite,(loss,lr,grad_norm))):
            raise ValueError("Non-finite loss, LR or gradient norm; stopped")
        if not rows:
            if set(camera_audit["forward"]) != set(CAMERAS) or set(camera_audit["backward"]) != set(CAMERAS):
                raise ValueError("Three-camera forward/backward evidence incomplete")
            camera_audit.update({"batch_shapes":{key:list(batch[key].shape)
                                                for key in ["observation.state","action",*CAMERAS]},
                                 "all_three_cameras_forward_and_backward":True})
            (cfg.output_dir/"camera_audit.json").write_text(json.dumps(camera_audit,indent=2)+"\n")
            print("THREE_CAMERA_FORWARD_BACKWARD_OK "+json.dumps(camera_audit),flush=True)
            metrics_file=(cfg.output_dir/"metrics.jsonl").open("w",buffering=1)
        step=tracker.steps+1
        row={"step":step,"train_loss":loss,"learning_rate":lr,"grad_norm":grad_norm,
             "wall_clock_s":time.perf_counter()-started,
             "training_loop_elapsed_s":time.perf_counter()-first_update_time,
             "step_s":float(tracker.update_s.val+tracker.dataloading_s.val),
             "gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
             "gpu_peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
        rows.append(row)
        metrics_file.write(json.dumps(row)+"\n")
        if step==1 or step%10==0:
            recent=rows[-10:]
            print(f"DIFFUSION_STATS step={step}/{cfg.steps} loss={np.mean([x['train_loss'] for x in recent]):.6f} "
                  f"lr={lr:.3g} steps_per_s={len(recent)/sum(x['step_s'] for x in recent):.3f} "
                  f"peak_alloc_mib={row['gpu_peak_allocated_mib']:.1f} "
                  f"peak_reserved_mib={row['gpu_peak_reserved_mib']:.1f} wall_s={row['wall_clock_s']:.1f}",flush=True)
        return result

    def save_with_alias(*args,**kwargs):
        result=original_save(*args,**kwargs)
        checkpoint=Path(kwargs["checkpoint_dir"])
        alias=checkpoint.parent/str(kwargs["step"])
        if alias!=checkpoint and not alias.exists():
            alias.symlink_to(checkpoint.name,target_is_directory=True)
        return result

    official.make_policy=checked_make_policy
    official.update_policy=measured_update
    official.save_checkpoint=save_with_alias
    error=None
    try:
        official.train(cfg)
        last=cfg.output_dir/"checkpoints/last/pretrained_model"
        missing=[name for name in REQUIRED_MODEL_FILES if not (last/name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete final model/processors: {missing}")
        shutil.copy2(last/"train_config.json",cfg.output_dir/"train_config.json")
    except BaseException as exc:
        error={"type":type(exc).__name__,"message":str(exc),"oom":isinstance(exc,torch.cuda.OutOfMemoryError)}
        raise
    finally:
        if metrics_file:
            metrics_file.close()
        for handle in handles:
            handle.remove()
        official.make_policy=original_make_policy
        official.update_policy=original_update
        official.save_checkpoint=original_save
        wall=time.perf_counter()-started
        summary={
            "completed_steps":len(rows),"requested_steps":cfg.steps,"batch_size":cfg.batch_size,
            "seed":cfg.seed,"error":error,"wall_clock_s_including_setup_and_checkpoint":wall,
            "training_loop_elapsed_s":rows[-1]["training_loop_elapsed_s"] if rows else None,
            "steady_state_steps_per_s_excluding_first_10":
                len(rows[10:])/sum(r["step_s"] for r in rows[10:]) if len(rows)>10 else None,
            "max_gpu_allocated_mib":max((r["gpu_peak_allocated_mib"] for r in rows),default=None),
            "max_gpu_reserved_mib":max((r["gpu_peak_reserved_mib"] for r in rows),default=None),
            "loss_first_50_mean":float(np.mean([r["train_loss"] for r in rows[:50]])) if rows else None,
            "loss_last_50_mean":float(np.mean([r["train_loss"] for r in rows[-50:]])) if rows else None,
            "three_camera_audit":camera_audit,"protocol":policy_protocol(cfg.policy),
            "spatial_behavior_evaluated":False,"smoke_is_not_a_trained_baseline":True if cfg.steps<=500 else False,
        }
        if cfg.output_dir.exists():
            (cfg.output_dir/"smoke_summary.json" if cfg.steps<=500 else cfg.output_dir/"training_summary.json").write_text(
                json.dumps(summary,indent=2)+"\n")
        print("RUN_SUMMARY "+json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--engineering-from-scratch",action="store_true",
                        help="Explicitly opt into the archived scratch DP engineering run; NOT RoboCasa pretrained fine-tuning")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.smoke and not args.preflight and not args.engineering_from_scratch:
        parser.error("This entry trains scratch LeRobot U-Net DP, not RoboCasa pretrained DP. "
                     "Use --smoke for engineering checks. Pretrained fine-tuning needs its native implementation; "
                     "see docs/robocasa_pretrained_dp.md.")
    os.chdir(ROOT)
    cfg=load_config(args)
    preflight(cfg)
    if args.preflight:
        return
    if args.worker:
        train_worker(cfg)
        return
    cfg.output_dir.parent.mkdir(parents=True,exist_ok=True)
    temporary_log=cfg.output_dir.parent/f".{cfg.output_dir.name}.train.log"
    if temporary_log.exists():
        raise FileExistsError(f"Preserving previous failed log: {temporary_log}")
    command=[sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    with temporary_log.open("w",buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,cwd=ROOT,env={**os.environ,"TQDM_DISABLE":"1"})
        try:
            for line in process.stdout:
                log.write(line)
                if ("DIFFUSION_" in line or "POLICY_CREATED" in line or "THREE_CAMERA_" in line
                    or "RUN_SUMMARY" in line or "Traceback" in line or "Error" in line
                    or "Checkpoint policy" in line):
                    if "DIFFUSION_STATS" in line:
                        line=line[line.index("DIFFUSION_STATS"):]
                    print(line.rstrip(),flush=True)
            return_code=process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if cfg.output_dir.exists():
        shutil.move(temporary_log,cfg.output_dir/"train.log")
    if return_code:
        raise SystemExit(f"Official Diffusion failed ({return_code}). OOM is not retried; see train.log.")

if __name__=="__main__":
    main()
