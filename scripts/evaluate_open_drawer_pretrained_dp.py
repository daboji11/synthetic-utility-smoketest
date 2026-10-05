#!/usr/bin/env python
"""Official native EMA DP: one pretrained or H50-left fine-tuned rollout."""
from __future__ import annotations
import argparse
from collections import deque
import gc
import json
from pathlib import Path
import time
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300"
REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
SNAPSHOT = ROOT / "data/raw/huggingface_models/models--robocasa--robocasa365_checkpoints/snapshots" / REVISION
DEFAULT_CHECKPOINT = SNAPSHOT / PREFIX / "checkpoints/latest.ckpt"

def load_native_policy(checkpoint, expected_updates=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    is_finetuned = cfg.task.name == "h50_open_drawer_left"
    if cfg.task.name not in ["pretrain_human300","h50_open_drawer_left"] or not cfg.training.use_ema:
        raise ValueError("Expected official pretrained EMA or its H50-left fine-tuned checkpoint")
    updates = int(dill.loads(payload["pickles"]["optimizer_updates"])) if is_finetuned else None
    if is_finetuned:
        required = expected_updates if expected_updates is not None else int(cfg.native_finetune.steps)
        if (updates != required or cfg.native_finetune.source_revision != REVISION
            or cfg.native_finetune.episodes != 28 or cfg.native_finetune.fps != 20):
            raise ValueError(f"Incomplete/incompatible H50 checkpoint: updates={updates}, required={required}")
    elif expected_updates is not None:
        raise ValueError("Optimizer-update gate applies only to fine-tuned checkpoints")
    policy = hydra.utils.instantiate(cfg.policy)
    loaded = policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    # Keep the checkpoint normalizer, including RGB/position/action transforms.
    state = policy.normalizer.state_dict()
    action_scale = state["params_dict.action.scale"]
    action_offset = state["params_dict.action.offset"]
    if not torch.equal(action_scale, torch.ones(12)) or not torch.equal(action_offset, torch.zeros(12)):
        raise ValueError("Unexpected non-identity pretrained delta-action normalization")
    companion = checkpoint.parent.parent / "normalizer.pkl"
    if not companion.is_file():
        raise FileNotFoundError(f"Missing official companion normalizer: {companion}")
    if companion.exists():
        with companion.open("rb") as f:
            saved = dill.load(f)
        companion_state = saved.state_dict()
        if set(state) != set(companion_state) or any(not torch.equal(state[k], companion_state[k]) for k in state):
            raise ValueError("EMA checkpoint and published companion normalizer differ")
    protocol = {
        "checkpoint": str(checkpoint.absolute()), "checkpoint_blob": str(checkpoint.resolve()), "checkpoint_revision": REVISION,
        "policy_class": cfg.policy._target_, "pretraining_task": "pretrain_human300", "checkpoint_task": cfg.task.name,
        "loaded_weights": "ema_model", "strict_load": True,
        "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
        "n_obs_steps": policy.n_obs_steps, "prediction_horizon": policy.horizon,
        "executed_actions_per_inference": policy.n_action_steps,
        "denoising_steps": policy.num_inference_steps,
        "normalizer_source": "strictly loaded EMA checkpoint (companion verified)",
        "action_normalization": "identity; no LeRobot processors or action scaling",
        "normalizer_fields": list(policy.normalizer.params_dict),
        "inference_dtype": "float32", "control_hz": 20,
        "language_encoder": "openai/clip-vit-large-patch14, robomimic LangEncoder",
        "source_is_pretrained": True, "fine_tuning_started": is_finetuned,
        "optimizer_updates": updates,
        "policy_language_text": str(cfg.native_finetune.language_text) if is_finetuned else None,
    }
    del payload
    gc.collect()
    policy = policy.to("cuda").eval()
    return policy, protocol

def process_native_observation(raw, shape_meta, lang_emb):
    # Exactly the official RobomimicImageWrapper conversion.
    from robomimic.utils.obs_utils import process_frame
    out = {}
    for key, meta in shape_meta["obs"].items():
        if key == "lang_emb":
            out[key] = lang_emb.copy()
            continue
        value = np.asarray(raw[meta["lerobot_keys"][0]])
        if meta.get("type") == "rgb":
            if value.dtype != np.uint8 or value.shape != (256,256,3):
                raise ValueError(f"Invalid RGB observation {key}: {value.shape}, {value.dtype}")
            value = process_frame(value, channel_dim=3, scale=255.)
        value = value.astype(np.float32)
        if value.shape != tuple(meta["shape"]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation {key}: {value.shape}")
        out[key] = value
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--split", choices=["all","pretrain","target","test"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT/"results/robocasa_eval/pretrained_dp_seed_3")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing output: {args.output}")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy, protocol = load_native_policy(args.checkpoint, args.expected_updates)
    print("CHECKPOINT_OK " + json.dumps(protocol), flush=True)
    from robomimic.utils.lang_utils import LangEncoder
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from robocasa.utils.env_utils import convert_action
    import cv2
    lang_encoder = LangEncoder("cpu")
    env = None
    writer = None
    started = time.monotonic()
    args.output.mkdir(parents=True)
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    try:
        env = RoboCasaGymEnv(env_name="OpenDrawer", split=args.split, control_freq=20,
                            camera_widths=256, camera_heights=256, obj_registries=("lightwheel",))
        raw, _ = env.reset(seed=args.seed)
        instruction = raw["annotation.human.task_description"]
        if instruction.strip().lower().rstrip(".") != "open the left drawer":
            raise ValueError(f"Seed={args.seed} is not left drawer: {instruction!r}")
        language_text = protocol["policy_language_text"] or instruction
        lang = lang_encoder.get_lang_emb(language_text).cpu().numpy()
        del lang_encoder
        if lang.shape != (768,) or env.env.control_freq != 20:
            raise ValueError("Language or control frequency mismatch")
        protocol.update({"seed":args.seed, "split":args.split, "instruction":instruction,
                         "actual_control_hz":env.env.control_freq, "actual_policy_language_text":language_text,
                         "env_metadata":env.env.get_ep_meta(),
                         "camera_names":list(env.env.camera_names)})
        (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
        camera_keys = [key for key,meta in protocol["shape_meta"]["obs"].items() if meta.get("type")=="rgb"]
        def write_frame(observation, step, reward, success):
            frames=[observation[protocol["shape_meta"]["obs"][key]["lerobot_keys"][0]] for key in camera_keys]
            canvas=np.concatenate(frames,axis=1)
            canvas=np.concatenate([canvas,np.zeros((32,768,3),dtype=np.uint8)],axis=0)
            cv2.putText(canvas,f"pretrained DP seed={args.seed} step={step} success={int(success)} reward={reward:.1f}",
                        (4,277),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
        writer=cv2.VideoWriter(str(args.output/"rollout.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(768,288))
        if not writer.isOpened():
            raise RuntimeError("Streaming VideoWriter failed")
        write_frame(raw,0,0,False)
        initial=process_native_observation(raw,protocol["shape_meta"],lang)
        history=deque([initial]*policy.n_obs_steps,maxlen=policy.n_obs_steps)
        actions=deque()
        all_actions=[]
        replan_steps=[]
        clipped=np.zeros(12,dtype=np.int64)
        max_drawer=0.0
        inference_s=0.0
        denoise_calls=[]
        hook=policy.model.register_forward_hook(lambda module,inputs,output:denoise_calls.append(1))
        print(f"ENV_INTERFACE_OK task=OpenDrawer instruction={instruction!r} cameras=3x256x256 "
              f"proprio=3+4+2 language=768 action=12 control_hz=20 chunk=8 seed={args.seed}",flush=True)
        with (args.output/"steps.jsonl").open("w",buffering=1) as step_log:
            for step in range(1,args.horizon+1):
                replanned=not actions
                if replanned:
                    observation={key:torch.from_numpy(np.stack([o[key] for o in history])).unsqueeze(0).to("cuda")
                                 for key in initial}
                    call_before=len(denoise_calls)
                    torch.cuda.synchronize()
                    infer_started=time.monotonic()
                    with torch.inference_mode():
                        prediction=policy.predict_action(observation)
                    torch.cuda.synchronize()
                    inference_s+=time.monotonic()-infer_started
                    chunk=prediction["action"].detach().cpu().numpy()[0]
                    if chunk.shape!=(8,12) or not np.isfinite(chunk).all() or len(denoise_calls)-call_before!=100:
                        raise ValueError("Invalid native action chunk/denoising execution")
                    actions.extend(chunk)
                    replan_steps.append(step)
                action=actions.popleft().copy()
                all_actions.append(action)
                clipped+=(np.abs(action)>1).astype(np.int64)
                applied=np.clip(action,-1,1)
                raw,reward,terminated,truncated,info=env.step(convert_action(applied))
                history.append(process_native_observation(raw,protocol["shape_meta"],lang))
                success=bool(info.get("success",False))
                drawer=env.env.drawer.get_door_state(env=env.env)
                max_drawer=max(max_drawer,*map(float,drawer.values()))
                step_log.write(json.dumps({"step":step,"success":success,"reward":float(reward),
                                           "replanned":replanned,"raw_action":action.tolist(),
                                           "drawer_state":{k:float(v) for k,v in drawer.items()}})+"\n")
                write_frame(raw,step,reward,success)
                if step==1 or step%100==0 or success:
                    print(f"PROGRESS seed={args.seed} step={step}/{args.horizon} replans={len(replan_steps)} "
                          f"success={success} reward={float(reward):.4f} clips={int(clipped.sum())} "
                          f"drawer_open={max_drawer:.4f} elapsed_s={time.monotonic()-started:.1f}",flush=True)
                if success or terminated or truncated:
                    break
        hook.remove()
        array=np.stack(all_actions)
        stats=[{"dim":i,"min":float(array[:,i].min()),"max":float(array[:,i].max()),
                "mean":float(array[:,i].mean()),"std":float(array[:,i].std()),
                "p99_abs":float(np.percentile(np.abs(array[:,i]),99)),"clipped_count":int(clipped[i])}
               for i in range(12)]
        summary={**protocol,"pipeline_passed":True,"steps":step,"success":success,"reward":float(reward),
                 "replans":len(replan_steps),"replan_steps":replan_steps,"denoiser_calls":len(denoise_calls),
                 "actions_finite":True,"clipped_count":int(clipped.sum()),"action_stats":stats,
                 "max_drawer_open":max_drawer,"wall_clock_s":time.monotonic()-started,
                 "inference_s":inference_s,"gpu_peak_allocated_mb":torch.cuda.max_memory_allocated()/1024**2,
                 "video":str(args.output/"rollout.mp4"),
                 "interpretation":"One fine-tuned H50-left episode" if protocol["fine_tuning_started"] else "One zero-shot interface smoke episode, not a success-rate estimate"}
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print("SUMMARY "+json.dumps(summary),flush=True)
    finally:
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()

if __name__=="__main__":
    main()
