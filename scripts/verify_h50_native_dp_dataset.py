#!/usr/bin/env python
"""Load H50 target data with the unmodified official dataset class; no optimizer/training."""
import json
import os
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256"
def main():
    os.environ["H50_NATIVE_DP_DATASET"]=str(DATASET)
    task=OmegaConf.load(ROOT/"configs/native_dp/task/h50_open_drawer_left.yaml")
    cfg=OmegaConf.create({"task":task,"horizon":10,"n_obs_steps":2})
    torch.set_num_threads(4)
    ds=hydra.utils.instantiate(cfg.task.dataset)
    if len(ds.datasets)!=1 or len(ds.datasets[0].trajectory_ids)!=28 or len(ds)!=5916:
        raise ValueError("Unexpected target dataset membership")
    base=ds.datasets[0]
    if base.delta_indices["action.end_effector_position"].tolist()!=list(range(-1,9)):
        raise ValueError("Unexpected action window")
    checked=0
    for idx,length in zip(base.trajectory_ids,base.trajectory_lengths):
        parquet=pq.read_table(DATASET/f"data/chunk-000/episode_{idx:06d}.parquet")
        states=np.asarray(parquet["observation.state"].to_pylist(),dtype=np.float32)
        actions=np.asarray(parquet["action"].to_pylist(),dtype=np.float32)
        for t in [0,int(length//2),int(length-1)]:
            sample=base[int(base.start_indices[base.get_trajectory_index(idx)]+t)]
            obs_idx=np.clip(np.array([t-1,t]),0,length-1)
            act_idx=np.clip(np.arange(t-1,t+9),0,length-1)
            if not np.array_equal(sample["action"].numpy(),actions[act_idx]):
                raise ValueError(f"Action order/padding mismatch: ep={idx} t={t}")
            for key,sl in [("robot0_base_to_eef_pos",slice(0,3)),
                           ("robot0_base_to_eef_quat",slice(3,7)),("robot0_gripper_qpos",slice(7,9))]:
                if not np.array_equal(sample["obs"][key].numpy(),states[obs_idx,sl]):
                    raise ValueError(f"State order mismatch: {key}")
            for key in ["robot0_agentview_left_image","robot0_agentview_right_image","robot0_eye_in_hand_image"]:
                value=sample["obs"][key]
                if value.shape!=(2,3,256,256) or value.min()<0 or value.max()>1 or not torch.isfinite(value).all():
                    raise ValueError("Invalid camera window")
            if sample["obs"]["lang_emb"].shape!=(2,768) or not torch.isfinite(sample["obs"]["lang_emb"]).all():
                raise ValueError("Invalid language")
            checked+=1
    normalizer=ds.get_normalizer()
    norm=normalizer.state_dict()
    action_identity=torch.equal(norm["params_dict.action.scale"],torch.ones(12)) and torch.equal(
        norm["params_dict.action.offset"],torch.zeros(12))
    # Check the actual pretrained policy accepts a native training batch, without backward or optimizer.
    from evaluate_open_drawer_pretrained_dp import load_native_policy, DEFAULT_CHECKPOINT
    policy,_=load_native_policy(DEFAULT_CHECKPOINT)
    batch={"obs":{key:value.unsqueeze(0).to("cuda") for key,value in sample["obs"].items()},
           "action":sample["action"].unsqueeze(0).to("cuda")}
    with torch.inference_mode():
        forward_loss=float(policy.compute_loss(batch))
    if not np.isfinite(forward_loss):
        raise ValueError("Nonfinite pretrained forward-only target loss")
    pretrained_norm=policy.normalizer.state_dict()
    normalizer_delta={}
    for key in ["robot0_base_to_eef_pos","robot0_gripper_qpos"]:
        normalizer_delta[key]={
            "pretrained_scale":pretrained_norm[f"params_dict.{key}.scale"].cpu().tolist(),
            "target_only_scale":norm[f"params_dict.{key}.scale"].cpu().tolist(),
            "pretrained_offset":pretrained_norm[f"params_dict.{key}.offset"].cpu().tolist(),
            "target_only_offset":norm[f"params_dict.{key}.offset"].cpu().tolist()}
    report={"official_dataset_class":type(ds).__module__+"."+type(ds).__name__,
            "episodes":28,"frames":len(ds),"fps":20,"instruction":"open the left drawer",
            "drawer_side":{"left":28,"right":0},"windows_checked":checked,
            "window_shapes":{key:list(value.shape) for key,value in sample["obs"].items()},
            "action_shape":list(sample["action"].shape),"action_identity_normalizer":action_identity,
            "state_and_action_order_verified":True,"boundary_padding":"official first/last repetition",
            "image_resolution":[256,256],"source_resolution":[128,128],
            "image_transform":"bilinear upsample only; source detail remains 128x128",
            "fine_tuning_started":False,"task_config_load_passed":True,
            "pretrained_forward_only_loss":forward_loss,"backward_or_optimizer_executed":False,
            "normalizer_comparison":normalizer_delta,
            "training_ready":False,
            "remaining_training_decisions":["fresh optimizer initialization from EMA",
                "preserve pretrained observation normalizer versus refit on 28 episodes",
                "fine-tuning learning rate and checkpoint schedule"],
            "language_annotation":"H50 literal lowercase instruction; env emits capitalized sentence with period"}
    (DATASET/"official_reader_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"records/h50_native_dp_dataset_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print("OFFICIAL_DATASET_OK "+json.dumps(report),flush=True)
if __name__=="__main__":
    main()
