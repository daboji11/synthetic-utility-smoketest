#!/usr/bin/env python
"""Explicitly started adapter-only pi05 training using official JAX train_step/checkpoint utilities."""
import argparse,dataclasses,importlib.util,json,os,time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYNPUT_BACKEND","dummy")
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
from pi05_h50_config import make_config,verify_dataset,verify_source,ROOT,CHECKPOINT,DATASET
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--train",action="store_true",help="Required to start optimizer updates")
 p.add_argument("--batch-size",type=int,required=True);p.add_argument("--steps",type=int,default=10000)
 p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--seed",type=int,default=0)
 p.add_argument("--exp-name",default="seed_0");p.add_argument("--num-workers",type=int,default=4)
 p.add_argument("--lr-schedule-steps",type=int,default=None,help="Keep the original LR schedule when stopping training early")
 p.add_argument("--save-interval",type=int,default=2500);p.add_argument("--keep-period",type=int,default=2500)
 a=p.parse_args()
 if not a.train:p.error("Use preflight_pi05_h50.py for checks; --train is required to start.")
 if a.steps<2 or a.batch_size<1 or a.lr<=0:p.error("Positive steps, batch and learning rate required")
 if (a.lr_schedule_steps is not None and a.lr_schedule_steps<a.steps) or a.save_interval<1 or a.keep_period<1:
  p.error("LR schedule must cover all updates; save interval and keep period must be positive")
 source=verify_source();verify_dataset()
 if not (CHECKPOINT/"source_manifest.json").exists():raise FileNotFoundError("Run download_pi05_robocasa.py first; complete pinned weights required")
 cfg=make_config(batch_size=a.batch_size,steps=a.steps,lr=a.lr,seed=a.seed,exp_name=a.exp_name,num_workers=a.num_workers,
                 lr_schedule_steps=a.lr_schedule_steps,save_interval=a.save_interval,keep_period=a.keep_period)
 if cfg.checkpoint_dir.parent.exists() and cfg.checkpoint_dir.exists():raise FileExistsError(f"Preserving {cfg.checkpoint_dir}")
 import jax,jax.numpy as jnp
 from flax import nnx
 from flax.traverse_util import flatten_dict
 from openpi.training import checkpoints,data_loader,sharding
 if not any(d.platform=="gpu" for d in jax.devices()):raise RuntimeError("CUDA JAX device required")
 spec=importlib.util.spec_from_file_location("synutility_official_pi05_train",source/"scripts/train.py")
 upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
 mesh=sharding.make_mesh(cfg.fsdp_devices)
 data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
 repl=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 if cfg.batch_size % jax.device_count():raise ValueError("Batch size must be divisible by GPU count")
 manager,_=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=cfg.keep_period,overwrite=False,resume=False)
 run=cfg.checkpoint_dir
 (run/"native_config_snapshot.json").write_text(json.dumps(dataclasses.asdict(cfg),indent=2,default=lambda x: x.tolist() if hasattr(x,"tolist") else str(x))+"\n")
 (run/"actual_config.json").write_text(json.dumps({**vars(a),"source_commit":__import__("pi05_h50_config").SOURCE_COMMIT,
 "checkpoint":str(CHECKPOINT),"dataset":str(DATASET),"episodes":28,"frames":5916,"fps":20,
 "model":dataclasses.asdict(cfg.model),"only_lora":True,"ema_decay":None,"normalization":"native pretrained z-score stats; unchanged",
 "checkpoint_dir":str(cfg.checkpoint_dir),"normalizer_sha256":__import__("hashlib").sha256((CHECKPOINT/"assets/norm_stats.json").read_bytes()).hexdigest()},indent=2)+"\n")
 started=time.perf_counter();load_start=time.perf_counter()
 loader=data_loader.create_data_loader(cfg,sharding=data_sharding,shuffle=True);iterator=iter(loader);batch=next(iterator)
 batch_wait_s=time.perf_counter()-load_start
 rng=jax.random.key(cfg.seed);train_rng,init_rng=jax.random.split(rng)
 state,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
 trainables=flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep="/")
 if not trainables or any("lora" not in k for k in trainables):raise ValueError("Unexpected non-LoRA trainable parameters")
 (run/"trainable_parameters.json").write_text(json.dumps({k:list(v.shape) for k,v in trainables.items()},indent=2)+"\n")
 step_fn=jax.jit(__import__("functools").partial(upstream.train_step,cfg),in_shardings=(repl,state_sharding,data_sharding),out_shardings=(state_sharding,repl),donate_argnums=(1,))
 print(f"PI05_LORA_START batch={a.batch_size} steps={a.steps} lr={a.lr} lr_schedule_steps={a.lr_schedule_steps or a.steps} save_interval={a.save_interval} trainable={sum(v.size for v in trainables.values())} output={run}",flush=True)
 lr_fn=cfg.lr_schedule.create()
 with (run/"metrics.jsonl").open("x") as metrics:
  for step in range(1,cfg.num_train_steps+1):
   t=time.perf_counter()
   with sharding.set_mesh(mesh):state,info=step_fn(train_rng,state,batch)
   info=jax.device_get(info)
   if any(not __import__("numpy").isfinite(v).all() for v in info.values()):raise FloatingPointError(f"Non-finite training metrics at step {step}")
   memory=jax.devices()[0].memory_stats() or {}
   row={"step":step,"train_loss":float(info["loss"]),"learning_rate":float(lr_fn(step-1)),
       "compute_seconds":time.perf_counter()-t,"data_wait_s":batch_wait_s,"wall_clock_s":time.perf_counter()-started,
       "gpu_bytes_in_use":memory.get("bytes_in_use"),"gpu_peak_bytes_in_use":memory.get("peak_bytes_in_use"),
       **{k:float(v) for k,v in info.items() if k!="loss"}}
   metrics.write(json.dumps(row)+"\n")
   if step<=3 or step%cfg.log_interval==0:metrics.flush();print("TRAIN_STEP "+json.dumps(row),flush=True)
   if step%cfg.save_interval==0 or step==cfg.num_train_steps:
    checkpoints.save_state(manager,state,loader,step);manager.wait_until_finished()
    last=cfg.checkpoint_dir/"last"
    if last.is_symlink():last.unlink()
    elif last.exists():raise FileExistsError(f"Preserving non-symlink {last}")
    last.symlink_to(str(step),target_is_directory=True)
    print(f"CHECKPOINT_SAVED step={step} path={cfg.checkpoint_dir/str(step)}",flush=True)
   if step<cfg.num_train_steps:
    wait_start=time.perf_counter();batch=next(iterator);batch_wait_s=time.perf_counter()-wait_start
    if step<=3 or step%cfg.log_interval==0:print(f"DATA_WAIT seconds={time.perf_counter()-wait_start:.6f}",flush=True)
 manager.wait_until_finished()
 (run/"training_summary.json").write_text(json.dumps({"updates":cfg.num_train_steps,"final_checkpoint":str(cfg.checkpoint_dir/"last"),"wall_clock_s":time.perf_counter()-started,"batch_size":cfg.batch_size,"only_lora":True},indent=2)+"\n")
 print("PI05_LORA_COMPLETE",flush=True)
if __name__=="__main__":main()
