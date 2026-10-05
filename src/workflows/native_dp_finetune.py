"""Minimal target-data/initialization/save adapters around the unchanged official DP trainer."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import pickle
import time
import dill
import numpy as np
import torch
from diffusion_policy.dataset.lerobot_dataset import LerobotCotrainingDataset
from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace

def normalizer_equal(left,right):
    a,b=left.state_dict(),right.state_dict()
    return set(a)==set(b) and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)

class PretrainedNormalizerH50Dataset(LerobotCotrainingDataset):
    """Keep all official window/action/camera loading; preserve the source normalizer."""
    def __init__(self,pretrained_normalizer_path,**kwargs):
        self.pretrained_normalizer_path=Path(pretrained_normalizer_path)
        paths=kwargs.get("dataset_paths")
        if kwargs.get("dataset_soup") is not None or not paths or len(paths)!=1:
            raise ValueError("Fine-tuning must use exactly the fixed H50 target dataset")
        path=Path(paths[0]).resolve()
        info=json.loads((path/"meta/info.json").read_text())
        episodes=[json.loads(row) for row in (path/"meta/episodes.jsonl").read_text().splitlines()]
        manifest=json.loads((path/"provenance_manifest.json").read_text())
        if (info["fps"]!=20 or len(episodes)!=28 or sum(ep["length"] for ep in episodes)!=5916
            or any(ep["tasks"]!=["open the left drawer"] for ep in episodes)
            or manifest["source_repo"]!="DAVIAN-Robotics/robocasa-H50"
            or manifest["source_type"]!="human" or manifest["resampling"]!="none; all original timesteps retained"):
            raise ValueError("Unexpected dataset membership/provenance/temporal convention")
        super().__init__(**kwargs)
        if len(self.datasets)!=1 or len(self)!=5916:
            raise ValueError("Official reader membership mismatch")

    def get_normalizer(self,**kwargs):
        with self.pretrained_normalizer_path.open("rb") as f:
            normalizer=pickle.load(f)
        state=normalizer.state_dict()
        if (not torch.equal(state["params_dict.action.scale"],torch.ones(12))
            or not torch.equal(state["params_dict.action.offset"],torch.zeros(12))):
            raise ValueError("Expected identity pretrained delta-action normalization")
        return copy.deepcopy(normalizer)

class H50PretrainedFineTuneWorkspace(TrainDiffusionTransformerHybridWorkspace):
    include_keys=["global_step","epoch","optimizer_updates"]

    def __init__(self,cfg,output_dir=None):
        if cfg.training.resume or cfg.task.get("ckpt_path") is not None or not cfg.training.use_ema:
            raise ValueError("Use fresh adaptation from EMA, not optimizer/epoch resume")
        if cfg.training.gradient_accumulate_every!=1:
            raise ValueError("Current verified protocol uses gradient accumulation=1")
        super().__init__(cfg,output_dir=output_dir)
        source=Path(cfg.native_finetune.source_checkpoint)
        payload=torch.load(source,map_location="cpu",pickle_module=dill,weights_only=False)
        if payload["cfg"].task.name!="pretrain_human300" or not payload["cfg"].training.use_ema:
            raise ValueError("Expected published pretrain_human300 EMA source")
        # Load EMA into BOTH models. The freshly constructed optimizer stays empty.
        self.model.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
        self.ema_model.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
        with Path(cfg.native_finetune.pretrained_normalizer_path).open("rb") as f:
            saved_normalizer=pickle.load(f)
        if not normalizer_equal(self.model.normalizer,saved_normalizer):
            raise ValueError("Published checkpoint/normalizer mismatch")
        if self.optimizer.state or self.global_step!=0 or self.epoch!=0:
            raise ValueError("Optimizer/counters must start fresh")
        self.optimizer_updates=0
        self._last_loss=None
        self._metrics_file=None
        self._started=None
        self._first_gradient_report=None
        del payload

    def _after_optimizer_step(self,optimizer,args,kwargs):
        self.optimizer_updates+=1
        if self._last_loss is None or not math.isfinite(self._last_loss):
            raise ValueError("Invalid loss")
        if self.optimizer_updates==1:
            gradients={}
            for camera in ["robot0_agentview_right_image","robot0_agentview_left_image","robot0_eye_in_hand_image"]:
                params=self.model.obs_encoder.obs_nets[camera].parameters()
                values=[float(p.grad.detach().float().norm().item()) for p in params if p.grad is not None]
                gradients[camera]={"nonzero_gradient":any(v>0 for v in values),"gradient_norm_sum":sum(values)}
                if not values or not all(math.isfinite(v) for v in values) or not any(v>0 for v in values):
                    raise ValueError(f"Invalid camera gradient: {camera}")
            self._first_gradient_report=gradients
            print("CAMERA_GRADIENTS "+json.dumps(gradients),flush=True)
        record={"step":self.optimizer_updates,"train_loss":self._last_loss,
                "learning_rate":float(optimizer.param_groups[0]["lr"]),
                "gpu_allocated_mb":torch.cuda.memory_allocated()/1024**2,
                "gpu_peak_allocated_mb":torch.cuda.max_memory_allocated()/1024**2,
                "gpu_peak_reserved_mb":torch.cuda.max_memory_reserved()/1024**2,
                "wall_clock_s":time.monotonic()-self._started}
        self._metrics_file.write(json.dumps(record)+"\n")
        if self.optimizer_updates<=3 or self.optimizer_updates%100==0:
            print("TRAIN_STEP "+" ".join(f"{k}={v:.6f}" if isinstance(v,float) else f"{k}={v}" for k,v in record.items()),flush=True)

    def save_checkpoint(self,path=None,tag="latest",exclude_keys=None,include_keys=None,use_thread=False):
        # The stock workspace requests saves at epoch boundaries. Use exact optimizer counts.
        updates=self.optimizer_updates
        interval=int(self.cfg.native_finetune.save_freq)
        if updates==0 or (updates%interval!=0 and updates!=self.cfg.native_finetune.steps):
            return None
        output=Path(self.output_dir)
        directory=output/"checkpoints"
        directory.mkdir(parents=True,exist_ok=True)
        saved=super().save_checkpoint(
            path=directory/f"{updates:06d}.ckpt",exclude_keys=exclude_keys,include_keys=include_keys,use_thread=False)
        for name in ["last.ckpt","latest.ckpt"]:
            link=directory/name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(Path(saved).name)
        (output/"checkpoint_status.json").write_text(json.dumps({
            "optimizer_updates":updates,"checkpoint":saved,"weights":"EMA and train model",
            "normalizer":str(output/"normalizer.pkl"),
            "complete":updates==self.cfg.native_finetune.steps},indent=2)+"\n")
        print(f"CHECKPOINT_SAVED step={updates} path={saved} last={directory/'last.ckpt'}",flush=True)
        return saved

    def run(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        output=Path(self.output_dir)
        output.mkdir(parents=True,exist_ok=False)
        from omegaconf import OmegaConf
        OmegaConf.save(self.cfg,output/"config.yaml",resolve=True)
        (output/"config.json").write_text(json.dumps(OmegaConf.to_container(self.cfg,resolve=True),indent=2)+"\n")
        self._metrics_file=(output/"metrics.jsonl").open("w",buffering=1)
        self._started=time.monotonic()
        original_forward=self.model.forward
        def finite_forward(batch):
            loss=original_forward(batch)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite training loss; refusing backward")
            self._last_loss=float(loss.detach().item())
            return loss
        self.model.forward=finite_forward
        hook=self.optimizer.register_step_post_hook(self._after_optimizer_step)
        try:
            super().run()
            if self.optimizer_updates!=self.cfg.native_finetune.steps:
                raise RuntimeError(f"Expected {self.cfg.native_finetune.steps} updates, got {self.optimizer_updates}")
            # Guarantees final checkpoint even for an epoch budget shorter than stock save interval.
            self.save_checkpoint()
            (output/"summary.json").write_text(json.dumps({
                "status":"complete","optimizer_updates":self.optimizer_updates,
                "official_global_step":self.global_step,"epochs":self.epoch,
                "source_checkpoint":self.cfg.native_finetune.source_checkpoint,
                "initialization":"published EMA -> model and EMA; fresh optimizer/counters",
                "normalizer":"published pretrained normalizer preserved",
                "dataset":self.cfg.task.dataset.dataset_paths[0],"episodes":28,"fps":20,
                "batch_size":self.cfg.dataloader.batch_size,"seed":self.cfg.training.seed,
                "final_loss":self._last_loss,"wall_clock_s":time.monotonic()-self._started,
                "camera_gradients_first_step":self._first_gradient_report,
                "checkpoint":str(output/"checkpoints/last.ckpt")},indent=2)+"\n")
            print(f"FINETUNE_COMPLETE updates={self.optimizer_updates} output={output}",flush=True)
        except BaseException as exc:
            (output/"failure.json").write_text(json.dumps({
                "optimizer_updates":self.optimizer_updates,"error":str(exc),"status":"failed"},indent=2)+"\n")
            raise
        finally:
            hook.remove()
            self.model.forward=original_forward
            self._metrics_file.close()
