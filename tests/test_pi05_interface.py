"""Small contract tests; never load weights, train or step a simulator."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from pi05_h50_config import observation
class Pi05InterfaceTests(unittest.TestCase):
 def raw(self):
  return {"state.end_effector_position_relative":np.arange(3),
   "state.end_effector_rotation_relative":np.arange(3,7),
   "state.gripper_qpos":np.arange(7,9),"state.base_position":np.arange(9,12),
   "state.base_rotation":np.arange(12,16),
   "video.robot0_agentview_left":np.full((256,256,3),11,np.uint8),
   "video.robot0_agentview_right":np.full((256,256,3),22,np.uint8),
   "video.robot0_eye_in_hand":np.full((256,256,3),33,np.uint8)}
 def test_native_state_order_and_distinct_cameras(self):
  obs=observation(self.raw())
  np.testing.assert_array_equal(obs["observation/state"],[0,1,2,3,4,5,6,9,10,11,12,13,14,15,7,8])
  for key,value in [("observation/image",11),("observation/right_image",22),("observation/wrist_image",33)]:
   self.assertTrue((obs[key]==value).all())
  self.assertEqual(obs["prompt"],"open the left drawer")
 def test_reject_float_camera(self):
  raw=self.raw();raw["video.robot0_agentview_left"]=raw["video.robot0_agentview_left"].astype(np.float32)
  with self.assertRaises(ValueError):observation(raw)
 def test_no_implicit_training_start(self):
  p=subprocess.run([sys.executable,str(ROOT/"scripts/finetune_pi05_h50_lora.py"),"--batch-size","1"],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn("--train is required",p.stderr)
 @unittest.skipUnless(importlib.util.find_spec("flax"),"Official pi05 env needed")
 def test_freeze_filter_does_not_convert_rng_state(self):
  import tempfile,json
  from flax import nnx
  from pi05_h50_config import make_config
  with tempfile.TemporaryDirectory() as td:
   assets=Path(td)/"assets";assets.mkdir()
   stats={k:{"mean":[0]*32,"std":[1]*32,"q01":[-1]*32,"q99":[1]*32} for k in ("state","actions")}
   (assets/"norm_stats.json").write_text(json.dumps({"norm_stats":stats}))
   cfg=make_config(checkpoint=Path(td))
   param=nnx.Param(np.zeros(1))
   self.assertTrue(cfg.freeze_filter(("PaliGemma","img","kernel"),param))
   self.assertFalse(cfg.trainable_filter(("PaliGemma","img","kernel"),param))
   self.assertTrue(cfg.trainable_filter(("PaliGemma","llm","attn","lora_a"),param))
   self.assertFalse(cfg.freeze_filter(("rng","count"),nnx.Variable(np.zeros(1,dtype=np.uint32))))
if __name__=="__main__":unittest.main()
