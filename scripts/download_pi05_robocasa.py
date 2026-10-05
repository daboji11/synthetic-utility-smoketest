#!/usr/bin/env python
"""Download only the pinned RoboCasa pi05 params/assets, never optimizer state or other models."""
import argparse,json,os
from pathlib import Path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"data/raw/models/pi05_pretrain_human300/75000"
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=DEFAULT);p.add_argument("--metadata-only",action="store_true");a=p.parse_args()
 from huggingface_hub import hf_hub_download
 m=json.loads((ROOT/"configs/pi05/checkpoint_manifest.json").read_text())
 for item in m["files"]:
  rel=Path(item["path"]).relative_to(m["prefix"])
  if a.metadata_only and rel.as_posix()!="assets/norm_stats.json":continue
  local=a.output/rel
  if local.exists() and local.stat().st_size==item["size"]:continue
  cached=hf_hub_download(m["repo_id"],item["path"],revision=m["revision"],cache_dir=ROOT/"data/raw/huggingface_models")
  local.parent.mkdir(parents=True,exist_ok=True)
  if local.exists():raise FileExistsError(f"Preserving incomplete/conflicting file: {local}")
  local.symlink_to(Path(cached).resolve())
 if not a.metadata_only:
  for item in m["files"]:
   local=a.output/Path(item["path"]).relative_to(m["prefix"])
   if not local.exists() or local.stat().st_size!=item["size"]:raise ValueError(f"Incomplete checkpoint: {local}")
  (a.output/"source_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
 print(json.dumps({"output":str(a.output),"metadata_only":a.metadata_only,"complete_weights":not a.metadata_only,"download_bytes":m["download_bytes"]}),flush=True)
if __name__=="__main__":main()
