#!/usr/bin/env python
# Read-only ACT/RoboCasa interface diagnosis for one fixed left-drawer seed.
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

import evaluate_open_drawer_h50_act_robocasa as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/robocasa_eval"
SEED = 3


def stats(data: np.ndarray, clipped: bool = False) -> dict:
    out = {}
    for i in range(12):
        values = data[:, i]
        abs_values = np.abs(values)
        row = {"min": float(values.min()), "max": float(values.max()),
               "mean": float(values.mean()), "std": float(values.std()),
               "mean_abs": float(abs_values.mean()),
               "p95_abs": float(np.quantile(abs_values, 0.95)),
               "p99_abs": float(np.quantile(abs_values, 0.99))}
        if clipped:
            n = int(np.count_nonzero(abs_values > 1))
            row.update({"clipped_count": n, "clipped_percentage": 100 * n / len(values)})
        out[f"dim_{i}"] = row
    return out


def load_training_actions():
    processed = base.DATASET / "data/chunk-000/file-000.parquet"
    train = np.asarray(pq.read_table(processed, columns=["action"])["action"].to_pylist(),
                       dtype=np.float32)
    manifest = json.loads((base.DATASET / "resampling_manifest.json").read_text())
    raw_snapshot = next((ROOT / "data/raw/huggingface/datasets--DAVIAN-Robotics--robocasa-H50/snapshots").iterdir())
    raw = pq.read_table(raw_snapshot / "data/chunk-000/file-000.parquet",
                        columns=["action"])["action"]
    eps = {row["episode_index"]: row for row in pq.read_table(
        raw_snapshot / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "tasks", "dataset_from_index", "length"]).to_pylist()}
    original = []
    cumulative_ratios = []
    for record in manifest["episodes"]:
        source = eps[record["source_episode_index"]]
        assert source["tasks"] == [base.LEFT_TASK]
        first = source["dataset_from_index"]
        length = source["length"]
        source_actions = np.asarray(raw.slice(first, length).to_pylist(), dtype=np.float32)
        selected_actions = source_actions[record["source_indices"]]
        original.append(source_actions)
        cumulative_ratios.append((np.abs(selected_actions).sum(axis=0),
                                  np.abs(source_actions).sum(axis=0)))
    original = np.concatenate(original)
    numerator = np.sum([item[0] for item in cumulative_ratios], axis=0)
    denominator = np.sum([item[1] for item in cumulative_ratios], axis=0)
    ratio = np.divide(numerator, denominator, out=np.full(12, np.nan), where=denominator>0)
    return train, original, ratio


def inspect_normalization(train):
    from safetensors import safe_open
    ckpt = base.CHECKPOINT
    pre_name = ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_name = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    with safe_open(pre_name, framework="pt") as f:
        pre_mean=f.get_tensor("action.mean").numpy()
        pre_std=f.get_tensor("action.std").numpy()
        image_mean=f.get_tensor(f"{base.CAMERA}.mean").numpy().reshape(-1)
    with safe_open(post_name, framework="pt") as f:
        post_mean=f.get_tensor("action.mean").numpy()
        post_std=f.get_tensor("action.std").numpy()
    return {"saved_preprocessor_loaded": pre_name.is_file(),
            "saved_postprocessor_loaded": post_name.is_file(),
            "action_mean_pre_post_match": bool(np.allclose(pre_mean,post_mean)),
            "action_std_pre_post_match": bool(np.allclose(pre_std,post_std)),
            "action_mean_matches_train": bool(np.allclose(pre_mean,train.mean(0),atol=1e-4)),
            "action_std_matches_train": bool(np.allclose(pre_std,train.std(0),atol=1e-4)),
            "saved_visual_mean_rgb_0_to_1": image_mean.tolist()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    video_path=OUT / "debug_seed_3.mp4"
    diagnostics_path=OUT / "action_diagnostics_seed_3.json"
    steps_path=OUT / "steps_seed_3.jsonl"
    table_path=OUT / "action_comparison_seed_3.csv"
    for p in (video_path,diagnostics_path,steps_path,table_path):
        if p.exists(): raise FileExistsError(f"Preserving existing diagnostic: {p}")
    train, original, cumulative_ratio = load_training_actions()
    features,policy,preprocessor,postprocessor=base.make_policy()
    norm=inspect_normalization(train)
    env=RoboCasaGymEnv(env_name="OpenDrawer",camera_widths=128,camera_heights=128,
                       split="all",obj_registries=("lightwheel",),control_freq=16)
    video=None
    actions=[]
    state_rows=[]
    chunks=[]
    max_drawer=0.0
    success=False
    reward=0.0
    started=time.monotonic()
    try:
        obs,_=env.reset(seed=SEED)
        instruction=env.env.get_ep_meta().get("lang","").strip().lower().rstrip(".")
        if instruction != base.LEFT_TASK: raise ValueError(f"Expected left drawer, got {instruction}")
        policy.reset()
        video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),16,(128,128))
        if not video.isOpened(): raise RuntimeError("VideoWriter failed to open")
        with steps_path.open("w",buffering=1) as log:
            for step in range(1,base.HORIZON+1):
                image=obs[base.RAW_CAMERA]
                if image.shape!=(128,128,3) or image.dtype!=np.uint8: raise ValueError("Camera mismatch")
                video.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                frame=base.obs_to_policy(obs,features)
                if step==1:
                    print("policy_input",{k:(tuple(v.shape),str(v.dtype),float(v.min()),float(v.max()))
                                           for k,v in frame.items() if isinstance(v,torch.Tensor)},flush=True)
                state_rows.append(frame["observation.state"].detach().cpu().numpy().reshape(-1))
                prepared=preprocessor(frame)
                queue_before=len(policy._action_queue)
                with torch.inference_mode():
                    predicted=policy.select_action(prepared)
                if queue_before==0:
                    chunks.append({"step":step,"predicted_actions":policy.config.chunk_size,
                                   "executed_actions":policy.config.n_action_steps})
                action=postprocessor(predicted).detach().cpu().numpy().reshape(-1)
                if action.shape!=(12,) or not np.isfinite(action).all(): raise ValueError("Invalid action")
                actions.append(action.copy())
                applied=np.clip(action,-1,1).astype(np.float32)
                obs,reward,terminated,truncated,info=env.step(convert_action(applied))
                success=bool(info.get("success",False))
                door_state=env.env.drawer.get_door_state(env=env.env)
                max_drawer=max(max_drawer,max(float(x) for x in door_state.values()))
                log.write(json.dumps({"step":step,"reward":float(reward),"success":success,
                                      "raw_action":action.tolist(),"applied_action":applied.tolist(),
                                      "door_state":{k:float(v) for k,v in door_state.items()}})+chr(10))
                if step%100==0: print(f"diagnostic seed={SEED} step={step}/{base.HORIZON} success={success}",flush=True)
                if success or terminated or truncated: break
            video.write(cv2.cvtColor(obs[base.RAW_CAMERA],cv2.COLOR_RGB2BGR))
    finally:
        if video is not None: video.release()
        env.close()
    rollout=np.asarray(actions,dtype=np.float32)
    train_stats=stats(train)
    original_stats=stats(original)
    rollout_stats=stats(rollout,clipped=True)
    with table_path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["dim","train_min","train_max","train_std","rollout_min","rollout_max",
                         "rollout_std","clipped_count","original20_mean_abs","train16_mean_abs",
                         "original20_p95_abs","train16_p95_abs","cumulative_abs_ratio_16_to_20"])
        for i in range(12):
            key=f"dim_{i}"
            writer.writerow([i,train_stats[key]["min"],train_stats[key]["max"],train_stats[key]["std"],
                             rollout_stats[key]["min"],rollout_stats[key]["max"],rollout_stats[key]["std"],
                             rollout_stats[key]["clipped_count"],original_stats[key]["mean_abs"],
                             train_stats[key]["mean_abs"],original_stats[key]["p95_abs"],
                             train_stats[key]["p95_abs"],None if np.isnan(cumulative_ratio[i]) else float(cumulative_ratio[i])])
    result={"seed":SEED,"steps":len(rollout),"success":success,"reward":float(reward),
            "video":str(video_path),"camera":base.CAMERA,"video_frames":len(rollout)+1,
            "env_control_freq":int(env.env.control_freq),"policy_select_action_calls":len(rollout),
            "policy_chunk_inference_calls":len(chunks),"chunk_events":chunks,
            "chunk_size":policy.config.chunk_size,"n_action_steps":policy.config.n_action_steps,
            "temporal_ensemble_coeff":policy.config.temporal_ensemble_coeff,
            "effective_dt_seconds":1/16,"max_drawer_open_fraction":max_drawer,
            "wall_clock_s":round(time.monotonic()-started,2),"normalization":norm,
            "action_dimensions":{f"dim_{i}":{"train16":train_stats[f"dim_{i}"],
                                         "original20":original_stats[f"dim_{i}"],
                                         "rollout_raw":rollout_stats[f"dim_{i}"],
                                         "cumulative_abs_ratio_16_to_20":None if np.isnan(cumulative_ratio[i]) else float(cumulative_ratio[i])}
                                 for i in range(12)}}
    diagnostics_path.write_text(json.dumps(result,indent=2)+chr(10))
    print("DIAGNOSTIC_COMPLETE",json.dumps({k:result[k] for k in ("seed","steps","success",
            "max_drawer_open_fraction","policy_chunk_inference_calls","video")}),flush=True)


if __name__=="__main__": main()
