#!/usr/bin/env python
"""User-started pi05 OpenDrawer rollout: native transforms, streamed three-view videos and explicit statistics."""
import argparse,collections,json,os,time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYNPUT_BACKEND","dummy")
from pi05_h50_config import make_config,observation,verify_source,CHECKPOINT,ROOT
SEEDS=[3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,default=CHECKPOINT)
 p.add_argument("--lora",action="store_true",help="Use adapter architecture for our finetuned checkpoints")
 p.add_argument("--seeds",type=int,nargs="+",default=[3]);p.add_argument("--episodes",type=int,choices=[1,3,20])
 p.add_argument("--horizon",type=int,default=750);p.add_argument("--replan-steps",type=int,default=5)
 p.add_argument("--output",type=Path);a=p.parse_args()
 if a.episodes:a.seeds=SEEDS[:a.episodes]
 if a.horizon<1 or not 1<=a.replan_steps<=50 or len(set(a.seeds))!=len(a.seeds):p.error("Invalid horizon/replan/seeds")
 import cv2,jax,numpy as np
 from openpi.policies import policy_config
 from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
 from robocasa.utils.env_utils import convert_action
 verify_source()
 if not any(d.platform=="gpu" for d in jax.devices()):raise RuntimeError("CUDA JAX device required")
 output=a.output or ROOT/"results/robocasa_eval"/("pi05_"+("lora" if a.lora else "vanilla")+"_"+time.strftime("%Y%m%d_%H%M%S"))
 if output.exists():raise FileExistsError(f"Preserving {output}")
 cfg=make_config(lora=a.lora)
 policy=policy_config.create_trained_policy(cfg,a.checkpoint)
 output.mkdir(parents=True)
 protocol={**vars(a),"checkpoint":str(a.checkpoint.resolve()),"output":str(output),"control_hz":20,
  "instruction":"open the left drawer","action_horizon":50,"state_order":list(__import__("pi05_h50_config").STATE_KEYS),
  "camera_keys":__import__("pi05_h50_config").CAMERAS,"normalization":"saved checkpoint assets; native transforms",
  "devices":[str(d) for d in jax.devices()],"source_commit":__import__("pi05_h50_config").SOURCE_COMMIT}
 (output/"eval_config.json").write_text(json.dumps(protocol,indent=2,default=str)+"\n")
 episodes=[]
 for seed in a.seeds:
  env=writer=None
  try:
   np.random.seed(seed)
   # Native policy holds a JAX PRNG key; reset it identically for paired vanilla/LoRA evaluations.
   policy._rng=jax.random.key(seed)
   env=RoboCasaGymEnv(env_name="OpenDrawer",split="all",control_freq=20,camera_widths=256,camera_heights=256,obj_registries=("lightwheel",))
   raw,_=env.reset(seed=seed)
   text=raw["annotation.human.task_description"]
   if text.strip().lower().rstrip(".")!="open the left drawer":raise ValueError(f"Seed {seed} is not a left-drawer task: {text}")
   if env.env.control_freq!=20:raise ValueError("Wrong control frequency")
   ep_dir=output/f"seed_{seed}";ep_dir.mkdir()
   writer=cv2.VideoWriter(str(ep_dir/"rollout.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(768,256))
   if not writer.isOpened():raise RuntimeError("Video writer failed")
   def frame(obs):
    rgb=np.concatenate([obs["video.robot0_agentview_left"],obs["video.robot0_agentview_right"],obs["video.robot0_eye_in_hand"]],axis=1)
    writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
   frame(raw);queue=collections.deque();replans=0;success=False;clips=0;actions=[];start=time.perf_counter()
   with (ep_dir/"steps.jsonl").open("w") as trace:
    for step in range(1,a.horizon+1):
     if not queue:
      chunk=np.asarray(policy.infer(observation(raw))["actions"])
      if chunk.shape!=(50,12) or not np.isfinite(chunk).all():raise ValueError(f"Invalid action chunk: {chunk.shape}")
      queue.extend(chunk[:a.replan_steps]);replans+=1
     action=queue.popleft();actions.append(action);clips+=int((np.abs(action)>1).sum())
     raw,reward,done,truncated,info=env.step(convert_action(np.clip(action,-1,1)))
     success=bool(info["success"]);frame(raw)
     trace.write(json.dumps({"step":step,"success":success,"reward":float(reward),"replans":replans,"action":action.tolist()})+"\n")
     if step==1 or step%100==0 or success:print(f"PROGRESS seed={seed} step={step}/{a.horizon} replans={replans} clips={clips} success={success} reward={reward} elapsed_s={time.perf_counter()-start:.1f}",flush=True)
     if success or done or truncated:break
   actions=np.asarray(actions)
   stats=[{"dim":i,"min":float(actions[:,i].min()),"max":float(actions[:,i].max()),"mean":float(actions[:,i].mean()),"std":float(actions[:,i].std()),"p99_abs":float(np.quantile(np.abs(actions[:,i]),.99)),"clipped":int((np.abs(actions[:,i])>1).sum())} for i in range(12)]
   row={"seed":seed,"success":success,"steps":step,"reward":float(reward),"replans":replans,"clipped_count":clips,"wall_clock_s":time.perf_counter()-start,"video":str(ep_dir/"rollout.mp4"),"action_stats":stats}
   (ep_dir/"summary.json").write_text(json.dumps(row,indent=2)+"\n");episodes.append(row)
   print("EPISODE_STATS "+json.dumps(row),flush=True)
   count=sum(e["success"] for e in episodes);print(f"PROGRESS episodes={len(episodes)}/{len(a.seeds)} successes={count} success_rate={count/len(episodes):.1%}",flush=True)
   (output/"episodes.json").write_text(json.dumps(episodes,indent=2)+"\n")
  finally:
   if writer is not None:writer.release()
   if env is not None:env.close()
 count=sum(e["success"] for e in episodes)
 report={"episodes":len(episodes),"successes":count,"success_rate":count/len(episodes),"seeds":a.seeds,"checkpoint":str(a.checkpoint),"episode_stats":episodes}
 (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n");print("FINAL_EVAL "+json.dumps({"episodes":len(episodes),"successes":count,"success_rate":report["success_rate"],"output":str(output)}),flush=True)
if __name__=="__main__":main()
