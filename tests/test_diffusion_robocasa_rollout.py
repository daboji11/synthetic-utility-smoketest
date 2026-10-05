"""Integration checks for the reused observation/action/video loop without simulator rollouts."""
from collections import deque
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from src.workflows.open_drawer_diffusion import CAMERAS
from src.workflows import robocasa_h50_rollout as rollout

def observation(step):
    return {
        "state.end_effector_position_relative":np.array([step,0,0],np.float32),
        "state.end_effector_rotation_relative":np.array([0,0,0,1],np.float32),
        "state.gripper_qpos":np.array([0,0],np.float32),
        "state.base_position":np.array([0,0,0],np.float32),
        "state.base_rotation":np.array([0,0,0,1],np.float32),
        **{rollout.raw_camera_key(key):np.full((128,128,3),[10+i,20+i,30+i],dtype=np.uint8)
           for i,key in enumerate(CAMERAS)},
    }

def test_loop_observes_every_step_replans32_streams_video_and_reports_gripper_clips(tmp_path,monkeypatch):
    original_obs_to_policy=rollout.obs_to_policy
    monkeypatch.setattr(rollout,"obs_to_policy",
        lambda obs,features:original_obs_to_policy(obs,features,device="cpu"))
    class Policy:
        config=SimpleNamespace(n_action_steps=32,use_amp=False)
        def reset(self):
            self._queues={"action":deque()}
            self.observed_steps=[]
            self.generations=0
        def select_action(self,batch):
            self.observed_steps.append(int(batch["observation.state"][0,0]))
            for i,key in enumerate(CAMERAS):
                torch.testing.assert_close(batch[key][0,:,0,0],
                                           torch.tensor([10+i,20+i,30+i],dtype=torch.float32)/255)
            if not self._queues["action"]:
                self.generations+=1
                action=torch.zeros(1,12)
                action[0,6]=1.1
                action[0,11]=-1
                self._queues["action"].extend([action]*32)
            return self._queues["action"].popleft()
    class Drawer:
        def get_door_state(self,env):
            return {"left":0.0}
    class Env:
        def __init__(self):
            self.env=SimpleNamespace(
                get_ep_meta=lambda:{"lang":"open the left drawer"},
                drawer=Drawer(),control_freq=20)
            self.action_space=SimpleNamespace(spaces={
                "action.end_effector_position","action.end_effector_rotation","action.gripper_close",
                "action.base_motion","action.control_mode"})
            self.steps=0
        def reset(self,seed):
            self.steps=0
            return observation(0),{}
        def step(self,action):
            self.steps+=1
            assert np.allclose(action["action.gripper_close"],1)
            assert np.allclose(action["action.control_mode"],-1)
            return observation(self.steps),0.0,False,False,{"success":False}
    policy=Policy()
    env=Env()
    result=rollout.run_episode(env,3,{key:None for key in CAMERAS},policy,
                              lambda x:x,lambda x:x,video_path=tmp_path/"video.mp4",
                              steps_path=tmp_path/"steps.jsonl",cameras=CAMERAS,control_hz=20,horizon=65)
    assert policy.observed_steps==list(range(65))
    assert policy.generations==3 and result["replan_steps"]==[1,33,65]
    assert result["steps"]==65
    assert result["arm_clipped_count"]==0
    assert result["gripper_clipped_count"]==65
    assert result["other_clipped_count"]==0
    assert result["action_stats_by_dim"][6]["clipped_pct"]==100
    assert len((tmp_path/"steps.jsonl").read_text().splitlines())==65
    video=cv2.VideoCapture(str(tmp_path/"video.mp4"))
    try:
        assert video.isOpened()
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))==66
        assert video.get(cv2.CAP_PROP_FPS)==20
        assert video.get(cv2.CAP_PROP_FRAME_WIDTH)==384
        assert video.get(cv2.CAP_PROP_FRAME_HEIGHT)==160
    finally:
        video.release()
