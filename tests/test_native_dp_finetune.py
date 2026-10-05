"""Test native trainer initialization/save bookkeeping with a tiny fixture, not a robot policy."""
import copy
import importlib.util
from pathlib import Path
import pickle
import sys
import dill
import pytest
import torch

if importlib.util.find_spec("diffusion_policy") is None:
    pytest.skip("Native DP environment required",allow_module_level=True)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from finetune_h50_native_dp import build_config
from evaluate_open_drawer_pretrained_dp import DEFAULT_CHECKPOINT,load_native_policy
from src.workflows.native_dp_finetune import H50PretrainedFineTuneWorkspace
from diffusion_policy.model.common.normalizer import LinearNormalizer,SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import hydra

CAMERAS=["robot0_agentview_right_image","robot0_agentview_left_image","robot0_eye_in_hand_image"]
class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder=torch.nn.Module()
        self.obs_encoder.obs_nets=torch.nn.ModuleDict({key:torch.nn.Linear(1,1,bias=False) for key in CAMERAS})
        self.normalizer=LinearNormalizer()
        self.normalizer["action"]=SingleFieldLinearNormalizer.create_manual(
            scale=torch.ones(12),offset=torch.zeros(12),input_stats_dict={})
        self.n_obs_steps=2
        self.horizon=10
        self.n_action_steps=8
        self.num_inference_steps=100
    @property
    def device(self):
        return next(self.parameters()).device
    def get_optimizer(self,learning_rate,**kwargs):
        return torch.optim.Adam(self.parameters(),lr=learning_rate)
    def set_normalizer(self,normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    def forward(self,batch):
        return sum(self.obs_encoder.obs_nets[key](batch["obs"][key]).square().mean() for key in CAMERAS)
    def predict_action(self,obs):
        size=next(iter(obs.values())).shape[0]
        return {"action_pred":torch.zeros(size,10,12,device=self.device)}

class TinyDataset(BaseImageDataset):
    def __init__(self,norm):
        self.norm=norm
    def __len__(self):
        return 4
    def __getitem__(self,index):
        return {"obs":{key:torch.ones(2,1) for key in CAMERAS},"action":torch.zeros(10,12)}
    def get_normalizer(self,**kwargs):
        return copy.deepcopy(self.norm)

@pytest.mark.skipif(not torch.cuda.is_available(),reason="Native stock loop fixture uses CUDA")
def test_fresh_ema_initialization_exact_updates_final_checkpoint_and_reload(tmp_path,monkeypatch):
    # Use the real config, but tiny weights/data to exercise the actual official training loop.
    cfg=build_config(DEFAULT_CHECKPOINT,tmp_path/"unused_dataset",tmp_path/"run",steps=2,batch_size=2)
    cfg.dataloader.num_workers=0
    cfg.logging.mode="disabled"
    source=TinyPolicy()
    for camera in CAMERAS:
        torch.nn.init.constant_(source.obs_encoder.obs_nets[camera].weight,2)
    source_payload={"cfg":copy.deepcopy(cfg),"state_dicts":{"ema_model":source.state_dict(),
                    "model":{key:value*0 for key,value in source.state_dict().items()},
                    "optimizer":{"invalid_pretrained_optimizer":"must not load"}},
                    "pickles":{"global_step":dill.dumps(999999),"epoch":dill.dumps(1999)}}
    source_payload["cfg"].task.name="pretrain_human300"
    checkpoint=tmp_path/"pretrained.ckpt"
    torch.save(source_payload,checkpoint,pickle_module=dill)
    normalizer=tmp_path/"normalizer.pkl"
    with normalizer.open("wb") as f:
        pickle.dump(source.normalizer,f)
    cfg.native_finetune.source_checkpoint=str(checkpoint)
    cfg.native_finetune.pretrained_normalizer_path=str(normalizer)
    instantiate_original=hydra.utils.instantiate
    def instantiate(config,*args,**kwargs):
        target=config.get("_target_","")
        if target.endswith("DiffusionTransformerHybridImagePolicy"):
            return TinyPolicy()
        if target.endswith("PretrainedNormalizerH50Dataset"):
            return TinyDataset(source.normalizer)
        return instantiate_original(config,*args,**kwargs)
    monkeypatch.setattr(hydra.utils,"instantiate",instantiate)
    workspace=H50PretrainedFineTuneWorkspace(cfg,output_dir=str(tmp_path/"run"))
    assert workspace.optimizer.state=={}
    assert workspace.global_step==workspace.epoch==workspace.optimizer_updates==0
    for key,value in source.state_dict().items():
        assert torch.equal(workspace.model.state_dict()[key],value)
        assert torch.equal(workspace.ema_model.state_dict()[key],value)
    workspace.run()
    import json
    output=tmp_path/"run"
    metrics=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics]==[1,2]
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in metrics)
    assert (output/"checkpoints/last.ckpt").resolve()==output/"checkpoints/000002.ckpt"
    saved=torch.load(output/"checkpoints/last.ckpt",map_location="cpu",pickle_module=dill,weights_only=False)
    assert dill.loads(saved["pickles"]["optimizer_updates"])==2
    assert set(saved["state_dicts"])=={"model","ema_model","optimizer"}
    assert (output/"normalizer.pkl").is_file()
    assert json.loads((output/"summary.json").read_text())["status"]=="complete"
    # Actual evaluator load path must accept complete native fine-tuned payloads and reject wrong update counts.
    policy,protocol=load_native_policy(output/"checkpoints/last.ckpt",expected_updates=2)
    assert protocol["fine_tuning_started"] and protocol["optimizer_updates"]==2
    assert protocol["policy_language_text"]=="open the left drawer"
    with pytest.raises(ValueError,match="Incomplete"):
        load_native_policy(output/"checkpoints/last.ckpt",expected_updates=10000)
