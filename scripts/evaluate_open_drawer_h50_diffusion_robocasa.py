#!/usr/bin/env python
"""Native20Hz three-camera Diffusion evaluation; no ACT configuration or checkpoint loading."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import time
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.workflows.open_drawer_diffusion import (
    BASE_OUTPUT, CAMERAS, DATASET, FPS, INSTRUCTION, REPO_ID, REQUIRED_MODEL_FILES,
    action_statistics, policy_protocol, validate_dataset,
)
from src.workflows.robocasa_h50_rollout import obs_to_policy, raw_camera_key, run_episode

def processor_state(checkpoint,filename,registry):
    config=json.loads((checkpoint/filename).read_text())
    steps=[x for x in config["steps"] if x["registry_name"]==registry]
    if len(steps)!=1 or not steps[0].get("state_file"):
        raise ValueError(f"Missing saved {registry} state")
    path=(checkpoint/steps[0]["state_file"]).resolve(strict=True)
    if not path.is_relative_to(checkpoint):
        raise ValueError("Processor state outside checkpoint")
    with safe_open(path,framework="pt",device="cpu") as f:
        return {k:f.get_tensor(k).numpy() for k in f.keys()},str(path)

def load_policy(checkpoint,require_step,device="cuda"):
    checkpoint=checkpoint.resolve(strict=True)
    if any(not (checkpoint/name).is_file() for name in REQUIRED_MODEL_FILES):
        raise FileNotFoundError("Incomplete official Diffusion checkpoint/processors")
    step=json.loads((checkpoint.parent/"training_state/training_step.json").read_text())["step"]
    if require_step is not None and step!=require_step:
        raise ValueError(f"Expected checkpoint step={require_step}, found {step}")
    dataset=validate_dataset()
    train=json.loads((checkpoint/"train_config.json").read_text())
    if (train["dataset"]["repo_id"]!=REPO_ID or train["policy"]["type"]!="diffusion"
        or train["dataset"].get("image_transforms",{}).get("enable",False)):
        raise ValueError("Checkpoint data/policy/transforms incompatible")
    # Root may differ when checkpoint is transferred from RunPod; fixed provenance and all saved stats verified below.
    config=DiffusionConfig.from_pretrained(str(checkpoint),local_files_only=True)
    if (list(config.image_features)!=CAMERAS
        or set(config.input_features)!={"observation.state",*CAMERAS}
        or tuple(config.input_features["observation.state"].shape)!=(16,)
        or any(tuple(config.image_features[key].shape)!=(3,128,128) for key in CAMERAS)
        or tuple(config.output_features["action"].shape)!=(12,)
        or not 1<=config.n_action_steps<=config.horizon-config.n_obs_steps+1):
        raise ValueError("Diffusion checkpoint schema/horizons mismatch")
    if config.normalization_mapping["ACTION"].value!="MIN_MAX" or config.normalization_mapping["STATE"].value!="MIN_MAX":
        raise ValueError("Expected official Diffusion min/max state and action normalization")
    pre,pre_file=processor_state(checkpoint,"policy_preprocessor.json","normalizer_processor")
    post,post_file=processor_state(checkpoint,"policy_postprocessor.json","unnormalizer_processor")
    for key in ["observation.state","action"]:
        for stat in ["min","max"]:
            if not np.allclose(pre[f"{key}.{stat}"],dataset.meta.stats[key][stat],rtol=1e-5,atol=1e-6):
                raise ValueError(f"Checkpoint {key}.{stat} differs from native20fps human base")
    for stat in ["min","max"]:
        if not np.array_equal(pre[f"action.{stat}"],post[f"action.{stat}"]):
            raise ValueError("Action pre/post normalization mismatch")
    for key in CAMERAS:
        for stat,imagenet in [("mean",[0.485,0.456,0.406]),("std",[0.229,0.224,0.225])]:
            expected=(np.asarray(imagenet).reshape(3,1,1) if train["dataset"].get("use_imagenet_stats",True)
                      else dataset.meta.stats[key][stat])
            if not np.allclose(pre[f"{key}.{stat}"],expected,rtol=1e-5,atol=1e-6):
                raise ValueError(f"Saved camera normalization mismatch: {key}.{stat}")
    config.device=device
    policy=DiffusionPolicy.from_pretrained(str(checkpoint),config=config,local_files_only=True,strict=True).eval()
    preprocessor,postprocessor=make_pre_post_processors(
        config,pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor":{"device":device}})
    validation={"resolved_checkpoint":str(checkpoint),"training_step":step,
                "saved_normalizer":pre_file,"saved_unnormalizer":post_file,
                "normalization_matches_native20fps_training":True,
                "camera_mapping":{key:raw_camera_key(key) for key in CAMERAS},
                "rgb_conversion":"RGB uint8 HWC -> float CHW/255 -> saved visual normalization",
                "state_order":"ee_pos3,ee_quat4,gripper_qpos2,base_pos3,base_quat4",
                "action_order":"ee_pos3,ee_rot3,gripper1,base_motion4,control_mode1",
                "no_action_scaling":True,
                "source_training_root":train["dataset"]["root"],"local_dataset":str(DATASET)}
    return dataset,policy,preprocessor,postprocessor,validation

def preflight(dataset,policy,preprocessor,postprocessor):
    sample=dataset[0]
    state=sample["observation.state"].numpy()
    raw={
        "state.end_effector_position_relative":state[:3],
        "state.end_effector_rotation_relative":state[3:7],
        "state.gripper_qpos":state[7:9],"state.base_position":state[9:12],
        "state.base_rotation":state[12:16],
    }
    for key in CAMERAS:
        image=sample[key]
        rgb=image if image.dtype==torch.uint8 else (image*255).round().to(torch.uint8)
        raw[raw_camera_key(key)]=rgb.permute(1,2,0).numpy()
    frame=obs_to_policy(raw,dataset.features,device=policy.config.device)
    if not np.array_equal(frame["observation.state"].cpu().numpy().reshape(-1),state):
        raise ValueError("Evaluation state ordering differs from training sample")
    for key in CAMERAS:
        expected=sample[key].float()/255 if sample[key].dtype==torch.uint8 else sample[key]
        if not torch.allclose(frame[key].cpu()[0],expected,atol=1e-6):
            raise ValueError(f"Evaluation camera conversion differs from training: {key}")
    prepared_training=preprocessor({**frame,"action":sample["action"].unsqueeze(0)})
    roundtrip=postprocessor(prepared_training["action"]).detach().cpu().reshape(-1)
    if not torch.allclose(roundtrip,sample["action"],atol=1e-6):
        raise ValueError("Action normalization roundtrip changed native H50 controls")
    # One official history/chunk inference, followed by cached actions; no simulator actions executed.
    policy.reset()
    calls=[]
    handle=policy.diffusion.unet.register_forward_hook(lambda module,inputs,output:calls.append(1))
    try:
        prepared=preprocessor(frame)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,
                                                  enabled=policy.config.use_amp and policy.config.device=="cuda"):
            action=postprocessor(policy.select_action(prepared))
            denoise_calls=len(calls)
            queued=len(policy._queues["action"])
            if queued!=policy.config.n_action_steps-1:
                raise ValueError("Diffusion action queue mismatch")
            for _ in range(queued):
                policy.select_action(preprocessor(frame))
            if len(calls)!=denoise_calls:
                raise ValueError("Cached action unexpectedly triggered diffusion inference")
        if action.shape!=(1,12) or not torch.isfinite(action).all():
            raise ValueError("Invalid unnormalized Diffusion action")
        if denoise_calls!=(policy.config.num_inference_steps or policy.config.num_train_timesteps):
            raise ValueError("Denoising step count mismatch")
        return {"camera_and_state_equal_saved_training_sample":True,"action_normalization_roundtrip_exact":True,
                "finite_postprocessed_action":True,"action_shape":list(action.shape),
                "actual_unet_calls_per_replan":denoise_calls,
                "actions_executed_per_replan":queued+1,
                "observation_queue_lengths":{key:len(value) for key,value in policy._queues.items()},
                "simulator_started":False}
    finally:
        handle.remove()
        policy.reset()

def training_action_stats(dataset):
    files=sorted((DATASET/"data").glob("**/*.parquet"))
    table=pa.concat_tables([pq.read_table(path,columns=["action"]) for path in files])
    return action_statistics(np.asarray(table["action"].to_pylist(),dtype=np.float32))

def save_comparison(output,result,train):
    path=output/f"action_comparison_seed_{result['seed']}.csv"
    columns=["dim","train_min","train_max","train_mean","train_std","train_p95_abs","train_p99_abs",
             "rollout_min","rollout_max","rollout_mean","rollout_std","rollout_p95_abs","rollout_p99_abs",
             "clipped_count"]
    with path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(columns)
        print("TRAIN_VS_ROLLOUT dim train_min train_max train_std rollout_min rollout_max rollout_std clipped",flush=True)
        for original,rollout in zip(train,result["action_stats_by_dim"],strict=True):
            writer.writerow([original["dim"],*[original[k] for k in ["raw_min","raw_max","raw_mean","raw_std","p95_abs","p99_abs"]],
                             *[rollout[k] for k in ["raw_min","raw_max","raw_mean","raw_std","p95_abs","p99_abs"]],rollout["clipped_count"]])
            print(f"{original['dim']:>2} {original['raw_min']: .3f} {original['raw_max']: .3f} "
                  f"{original['raw_std']:.3f} {rollout['raw_min']: .3f} {rollout['raw_max']: .3f} "
                  f"{rollout['raw_std']:.3f} {rollout['clipped_count']}",flush=True)
    (output/f"action_diagnostics_seed_{result['seed']}.json").write_text(
        json.dumps({"training":train,"rollout":result},indent=2)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=BASE_OUTPUT/"seed_0/checkpoints/last/pretrained_model")
    parser.add_argument("--require-training-step",type=int,default=10000)
    parser.add_argument("--episodes",type=int,default=3,choices=range(1,4))
    parser.add_argument("--seeds",nargs="+",type=int,default=[3,4,5])
    parser.add_argument("--start-seed",type=int,default=3)
    parser.add_argument("--find-left-seeds",action="store_true")
    parser.add_argument("--run-name",default="diffusion_seed_3_4_5")
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--preflight-device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    if (not args.run_name.isascii() or not args.run_name.replace("_","").isalnum()
        or args.start_seed<0 or (not args.find_left_seeds and (
            len(args.seeds)!=args.episodes or len(set(args.seeds))!=args.episodes or min(args.seeds)<0))):
        parser.error("Invalid run name/seeds; at most three episodes permitted")
    torch.set_num_threads(4)
    dataset,policy,preprocessor,postprocessor,validation=load_policy(args.checkpoint,args.require_training_step,device=args.preflight_device if args.preflight else "cuda")
    protocol={**policy_protocol(policy.config),"validation":validation,"seeds":None if args.find_left_seeds else args.seeds,
              "task":"OpenDrawer","instruction":INSTRUCTION,"requested_episodes":args.episodes,
              "horizon":get_task_horizon("OpenDrawer"),"split":"all","obj_registries":["lightwheel"],
              "versions":{name:importlib.metadata.version(name) for name in ["torch","lerobot","diffusers","robocasa","robosuite"]},
              "video_columns":CAMERAS,"action_clip":[-1,1],"no_ACT_checkpoint_or_processors":True}
    print("EVAL_PROTOCOL "+json.dumps(protocol),flush=True)
    if args.preflight:
        print("PREFLIGHT_OK "+json.dumps(preflight(dataset,policy,preprocessor,postprocessor)),flush=True)
        return
    eval_root=args.output_root or args.checkpoint.resolve().parents[2]/"robocasa_eval"
    output=eval_root/args.run_name
    if output.exists():
        raise FileExistsError(f"Preserving evaluation: {output}")
    output.mkdir(parents=True)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    train=training_action_stats(dataset)
    (output/"training_action_distribution.json").write_text(json.dumps(train,indent=2)+"\n")
    env=None
    results=[]
    skipped=[]
    started=time.monotonic()
    try:
        env=RoboCasaGymEnv(env_name="OpenDrawer",camera_widths=128,camera_heights=128,
                           split="all",obj_registries=("lightwheel",),control_freq=FPS)
        if env.env.control_freq!=FPS:
            raise ValueError("Actual control frequency is not native 20Hz")
        for key in CAMERAS:
            if tuple(env.observation_space[raw_camera_key(key)].shape)!=(128,128,3):
                raise ValueError(f"Missing actual camera {key}")
        protocol["actual_control_hz"]=env.env.control_freq
        protocol["actual_camera_names"]=list(env.env.camera_names)
        protocol["controller_types"]=[{key:type(value).__name__ for key,value in robot.composite_controller.part_controllers.items()}
                                      for robot in env.env.robots]
        (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
        print("ENV_INTERFACE_OK control_hz=20 camera_count=3 state_dim=16 action_dim=12",flush=True)
        with (output/"episodes.jsonl").open("w",buffering=1) as log:
            seed=args.start_seed
            while len(results)<args.episodes:
                if seed>=args.start_seed+100:
                    raise RuntimeError("Cannot find three accepted left-drawer seeds")
                current=seed if args.find_left_seeds else args.seeds[len(results)]
                result=run_episode(
                    env,current,dataset.features,policy,preprocessor,postprocessor,
                    video_path=output/"videos"/f"seed_{current}.mp4",
                    steps_path=output/"steps"/f"seed_{current}.jsonl",
                    cameras=CAMERAS,control_hz=FPS,horizon=protocol["horizon"])
                if result is None:
                    if not args.find_left_seeds:
                        raise ValueError(f"Requested seed={current} is not left drawer; no replacement")
                    skipped.append(current)
                    print(f"SKIPPED right drawer seed={current}",flush=True)
                else:
                    results.append(result)
                    log.write(json.dumps(result)+"\n")
                    save_comparison(output,result,train)
                    print(f"PROGRESS {len(results)}/{args.episodes} success={sum(x['success'] for x in results)} "
                          f"success_rate={sum(x['success'] for x in results)/len(results):.1%}",flush=True)
                seed+=1
    finally:
        if env is not None:
            env.close()
    summary={**protocol,"completed_episodes":len(results),"successes":sum(x["success"] for x in results),
             "success_rate":sum(x["success"] for x in results)/len(results),
             "selected_seeds":[x["seed"] for x in results],"skipped_non_left_seeds":skipped,
             "wall_clock_s":time.monotonic()-started,
             "spatial_behavior_assessment":"Inspect saved videos for approach, alignment, grasp and pull; no automatic diagnosis"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("SUMMARY "+json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
