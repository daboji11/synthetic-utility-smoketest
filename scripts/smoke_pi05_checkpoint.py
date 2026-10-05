#!/usr/bin/env python
"""Load the official checkpoint and infer once on H50 and a reset simulator observation; zero env steps."""
import json,os,time
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYNPUT_BACKEND","dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("OMP_NUM_THREADS","1")
from pi05_h50_config import make_config,verify_source,verify_dataset,observation,ROOT,CHECKPOINT
def main():
 import jax,numpy as np
 from openpi.policies import policy_config
 from openpi.training import data_loader
 from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
 source=verify_source();verify_dataset()
 if not any(d.platform=="gpu" for d in jax.devices()):raise RuntimeError("JAX CUDA required")
 cfg=make_config(lora=False)
 print("PI05_SMOKE loading vanilla official params",flush=True);t=time.perf_counter()
 policy=policy_config.create_trained_policy(cfg,CHECKPOINT)
 train_cfg=make_config(lora=True);dc=train_cfg.data.create(train_cfg.assets_dirs,train_cfg.model)
 ds=data_loader.create_torch_dataset(dc,train_cfg.model.action_horizon,train_cfg.model);item=ds[0]
 item.pop("actions")
 print("PI05_SMOKE infer H50 three-camera observation (first-call compile)",flush=True)
 actions=np.asarray(policy.infer(item)["actions"])
 if actions.shape!=(50,12) or not np.isfinite(actions).all():raise ValueError("Invalid H50 predicted chunk")
 env=None
 try:
  print("PI05_SMOKE reset OpenDrawer seed3; no env.step",flush=True)
  env=RoboCasaGymEnv(env_name="OpenDrawer",split="all",control_freq=20,camera_widths=256,camera_heights=256,obj_registries=("lightwheel",))
  raw,_=env.reset(seed=3)
  if raw["annotation.human.task_description"].strip().lower().rstrip(".")!="open the left drawer":raise ValueError("Wrong drawer side")
  sim_actions=np.asarray(policy.infer(observation(raw))["actions"])
  if sim_actions.shape!=(50,12) or not np.isfinite(sim_actions).all():raise ValueError("Invalid simulator predicted chunk")
  report={"passed":True,"source":str(source),"checkpoint":str(CHECKPOINT),"jax_version":jax.__version__,"devices":[str(d) for d in jax.devices()],
   "h50_action_shape":list(actions.shape),"sim_action_shape":list(sim_actions.shape),"h50_actions_finite":True,"sim_actions_finite":True,
   "h50_max_abs_action":float(np.abs(actions).max()),"sim_max_abs_action":float(np.abs(sim_actions).max()),
   "state_shape":[16],"camera_count":3,"control_hz":20,"seed":3,"env_steps_executed":0,"optimizer_updates":0,
   "success_rate_measured":False,"wall_clock_s":time.perf_counter()-t,"gpu_memory":jax.devices()[0].memory_stats()}
  (ROOT/"records/pi05_checkpoint_inference_smoke.json").write_text(json.dumps(report,indent=2)+"\n")
  print("PI05_CHECKPOINT_SMOKE_OK "+json.dumps(report),flush=True)
 finally:
  if env is not None:env.close()
if __name__=="__main__":main()
