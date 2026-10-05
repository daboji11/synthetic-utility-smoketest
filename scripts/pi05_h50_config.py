"""Pinned RoboCasa pi05 data/model configuration. No upstream source modifications."""
import dataclasses,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256"
CHECKPOINT=ROOT/"data/raw/models/pi05_pretrain_human300/75000"
SOURCE_COMMIT="5a6beda9ff99da30b4e1b59320f6a32971d7c397"
STATE_KEYS=("state.end_effector_position_relative","state.end_effector_rotation_relative",
            "state.base_position","state.base_rotation","state.gripper_qpos")
CAMERAS={"observation/image":"video.robot0_agentview_left",
         "observation/wrist_image":"video.robot0_eye_in_hand",
         "observation/right_image":"video.robot0_agentview_right"}
def verify_source():
 import subprocess,openpi
 source=Path(openpi.__file__).resolve().parents[2]
 actual=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
 if actual!=SOURCE_COMMIT:raise ValueError(f"openpi revision mismatch: {actual}")
 return source
def verify_dataset(dataset=DATASET):
 dataset=Path(dataset);info=json.loads((dataset/"meta/info.json").read_text());provenance=json.loads((dataset/"provenance_manifest.json").read_text())
 eps=[json.loads(x) for x in (dataset/"meta/episodes.jsonl").read_text().splitlines() if x.strip()]
 tasks=[json.loads(x) for x in (dataset/"meta/tasks.jsonl").read_text().splitlines() if x.strip()]
 if (info["fps"],info["total_episodes"],info["total_frames"])!=(20,28,5916):raise ValueError("Expected exactly 28 original-20Hz H50 left demos")
 if len(eps)!=28 or sum(e["length"] for e in eps)!=5916 or [t["task"] for t in tasks]!=["open the left drawer"]:raise ValueError("Wrong human base/instructions")
 if provenance["instruction_distribution"]!={"open the left drawer":28}:raise ValueError("Wrong source provenance")
 modality=json.loads((dataset/"meta/modality.json").read_text())
 expected={"end_effector_position_relative":(0,3),"end_effector_rotation_relative":(3,7),"gripper_qpos":(7,9),"base_position":(9,12),"base_rotation":(12,16)}
 for k,(start,end) in expected.items():
  if (modality["state"][k]["start"],modality["state"][k]["end"])!=(start,end):raise ValueError(f"Wrong state slices: {k}")
 return info
def observation(raw,prompt="open the left drawer"):
 result={k:np.ascontiguousarray(raw[v]) for k,v in CAMERAS.items()}
 for k,img in result.items():
  if img.dtype!=np.uint8 or img.ndim!=3 or img.shape[-1]!=3:raise ValueError(f"Expected RGB HWC uint8: {k}")
 result["observation/state"]=np.concatenate([raw[k] for k in STATE_KEYS]).astype(np.float32)
 if result["observation/state"].shape!=(16,) or not np.isfinite(result["observation/state"]).all():raise ValueError("Invalid 16D state")
 result["prompt"]=prompt
 return result
def make_config(*,lora=True,dataset=DATASET,checkpoint=CHECKPOINT,batch_size=1,steps=10000,lr=1e-5,seed=0,exp_name="seed_0",num_workers=4):
 from flax import nnx
 from openpi.models.pi0_config import Pi0Config
 from openpi.shared.nnx_utils import PathRegex
 from openpi.shared import normalize
 from openpi.training import config,optimizer,weight_loaders
 norm=normalize.load(Path(checkpoint)/"assets");base=config.get_config("pi05_pretrain_human300")
 model=Pi0Config(pi05=True,max_token_len=200,paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",action_expert_variant="gemma_300m_lora" if lora else "gemma_300m")
 # Native pretrained stats bypass the upstream H50-incompatible hardcoded stats reorder.
 data=config.LeRobotRobocasaDataConfig(data_dirs=[{"path":str(Path(dataset).resolve()),"filter_key":None}] if lora else [],base_config=config.DataConfig(norm_stats=norm,use_quantile_norm=False))
 return dataclasses.replace(base,name="pi05_h50_left_lora" if lora else "pi05_robocasa_vanilla",exp_name=exp_name,model=model,data=data,
    weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(checkpoint)/"params")),
    # Freeze ALL non-adapter weights including SigLIP and state/action heads.
    freeze_filter=nnx.All(nnx.Param,nnx.Not(PathRegex(".*lora.*"))),ema_decay=None,batch_size=batch_size,num_train_steps=steps,num_workers=num_workers,seed=seed,
    lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(500,max(1,steps//10)),peak_lr=lr,decay_steps=steps,decay_lr=lr/10),
    checkpoint_base_dir=str(ROOT/"outputs/OpenDrawer"),log_interval=10,save_interval=2500,keep_period=2500,wandb_enabled=False,
    policy_metadata={"task":"OpenDrawer","instruction":"open the left drawer","training_data":"28 H50 human left episodes","fps":20,"lora_only":lora})
