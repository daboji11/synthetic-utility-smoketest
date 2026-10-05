#!/usr/bin/env python
"""Check the real rendering/action interface before a long rental training job; no rollout."""
import json
import numpy as np
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action
def main():
    env=None
    try:
        env=RoboCasaGymEnv(env_name="OpenDrawer",split="all",control_freq=20,camera_widths=256,
                           camera_heights=256,obj_registries=("lightwheel",))
        raw,_=env.reset(seed=3)
        instruction=raw["annotation.human.task_description"]
        if instruction.strip().lower().rstrip(".")!="open the left drawer" or env.env.control_freq!=20:
            raise ValueError("Task/control convention mismatch")
        cameras=["video.robot0_agentview_right","video.robot0_agentview_left","video.robot0_eye_in_hand"]
        for key in cameras:
            if raw[key].shape!=(256,256,3) or raw[key].dtype!=np.uint8:
                raise ValueError(f"Invalid camera: {key}")
        for key,size in [("state.end_effector_position_relative",3),
                         ("state.end_effector_rotation_relative",4),("state.gripper_qpos",2)]:
            if raw[key].shape!=(size,) or not np.isfinite(raw[key]).all():
                raise ValueError(f"Invalid state: {key}")
        action=np.zeros(12,dtype=np.float32)
        action[6]=action[11]=-1
        if set(convert_action(action))!=set(env.action_space.spaces):
            raise ValueError("Action mapping mismatch")
        print("ROBOCASA_PREFLIGHT_OK "+json.dumps({"seed":3,"instruction":instruction,"control_hz":20,
              "camera_count":3,"camera_shape":[256,256,3],"proprio_dim":9,"action_dim":12,
              "rendering_and_assets_work":True,"env_steps_executed":0}),flush=True)
    finally:
        if env is not None:
            env.close()
if __name__=="__main__":
    main()
