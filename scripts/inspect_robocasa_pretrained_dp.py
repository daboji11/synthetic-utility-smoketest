#!/usr/bin/env python
"""Inspect the official pretrained RoboCasa DP metadata; optionally fetch ONLY latest weights."""
from pathlib import Path
import argparse
import json
import yaml
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REPO="robocasa/robocasa365_checkpoints"
REVISION="c484448aba1a9b60a04c9b0ca117241518ea69f3"
PREFIX="diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--download-weights",action="store_true",
                        help="Download the single approximately1.70GB latest.ckpt, not all epochs/models")
    args=parser.parse_args()
    cache=ROOT/"data/raw/huggingface_models"
    path=hf_hub_download(REPO,PREFIX+"/.hydra/config.yaml",revision=REVISION,cache_dir=cache)
    cfg=yaml.safe_load(Path(path).read_text())
    result={"repo_id":REPO,"revision":REVISION,"config_path":str(path),
            "policy_class":cfg["policy"]["_target_"],"pretraining_soup":cfg["task"]["dataset"]["dataset_soup"],
            "prediction_horizon":cfg["horizon"],"observation_horizon":cfg["n_obs_steps"],
            "action_horizon":cfg["n_action_steps"],"denoising_steps":cfg["policy"]["num_inference_steps"],
            "shape_meta":cfg["task"]["shape_meta"],"crop_shape":cfg["policy"]["crop_shape"],
            "directly_loadable_by_lerobot_smoke_trainer":False,"finetuning_ready":False}
    if args.download_weights:
        checkpoint=hf_hub_download(REPO,PREFIX+"/checkpoints/latest.ckpt",revision=REVISION,cache_dir=cache)
        result["checkpoint_path"]=str(checkpoint)
        result["normalizer_path"]=hf_hub_download(REPO,PREFIX+"/normalizer.pkl",revision=REVISION,cache_dir=cache)
        result["note"]="Files downloaded only; no pickle loading, policy integration or fine-tuning performed"
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
