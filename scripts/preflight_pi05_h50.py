#!/usr/bin/env python
"""No optimizer updates, no rollout. Check dataset/model transforms and adapter-only parameter shapes."""
import argparse,json
import numpy as np
from pi05_h50_config import make_config,verify_dataset,verify_source,ROOT
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--batch-size",type=int,default=1);a=p.parse_args()
 import jax
 from flax import nnx
 from openpi.training import data_loader
 source=verify_source();verify_dataset();cfg=make_config(batch_size=a.batch_size,num_workers=0)
 if not any(d.platform=="gpu" for d in jax.devices()):raise RuntimeError("CUDA JAX device required")
 dc=cfg.data.create(cfg.assets_dirs,cfg.model);raw=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model);sample=raw[0]
 for key in ("observation/image","observation/wrist_image","observation/right_image"):
  if sample[key].shape!=(256,256,3) or sample[key].dtype!=np.uint8:raise ValueError(f"Invalid camera {key}")
 if sample["actions"].shape!=(50,12) or sample["observation/state"].shape!=(16,):raise ValueError("Invalid state/action window")
 if sample["prompt"]!="open the left drawer":raise ValueError("Wrong task")
 import pyarrow.parquet as pq
 from pi05_h50_config import DATASET
 original=pq.read_table(DATASET/"data/chunk-000/episode_000000.parquet")
 source_state=np.asarray(original["observation.state"].to_pylist()[0],dtype=np.float32)
 reorder=list(range(7))+list(range(9,16))+[7,8]
 if not np.array_equal(sample["observation/state"],source_state[reorder]):raise ValueError("Official reader state reorder differs from H50")
 source_actions=np.asarray(original["action"].to_pylist()[:50],dtype=np.float32)
 if not np.array_equal(sample["actions"],source_actions):raise ValueError("Official reader changed action order/magnitudes")
 def model_shapes():
  m=cfg.model.create(jax.random.key(0));return nnx.state(m).filter(cfg.trainable_filter).to_pure_dict()
 from flax.traverse_util import flatten_dict
 flat=flatten_dict(jax.eval_shape(model_shapes),sep="/")
 if not flat or any("lora" not in k.lower() for k in flat):raise ValueError("Unexpected trainable parameters")
 transformed=data_loader.transform_dataset(raw,dc)[0]
 if set(transformed["image"])!={"base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"}:raise ValueError("Camera masks/slots mismatch")
 if any(x.shape!=(224,224,3) for x in transformed["image"].values()):raise ValueError("Invalid resized inputs")
 if not all(bool(x) for x in transformed["image_mask"].values()):raise ValueError("A camera was masked out")
 report={"passed":True,"source":str(source),"devices":[str(d) for d in jax.devices()],"episodes":28,"frames":5916,"fps":20,"batch_size":a.batch_size,
 "action_horizon":50,"action_dim":12,"state_dim":16,"trainable_parameter_count":sum(int(np.prod(x.shape)) for x in flat.values()),
 "trainable_parameters":{k:list(v.shape) for k,v in flat.items()},"state_reorder_matches_source":True,"action_order_magnitude_preserved":True,"all_three_camera_masks_true":True,"vision_frozen":True,"only_lora_trainable":True,"optimizer_updates":0,"full_model_weights_loaded":False,"rollout_executed":False}
 (ROOT/"records/pi05_preflight.json").write_text(json.dumps(report,indent=2)+"\n")
 print("PI05_PREFLIGHT_OK "+json.dumps({k:v for k,v in report.items() if k!="trainable_parameters"}),flush=True)
if __name__=="__main__":main()
