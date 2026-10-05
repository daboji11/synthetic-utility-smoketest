#!/usr/bin/env python
"""Evaluate a fixed set of H50-left native DP seeds with explicit stats and videos."""
import argparse
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/"outputs/OpenDrawer/dp_pretrained_human_left/seed_0_batch32"

def wilson_interval(successes,total):
    z=1.959963984540054
    p=successes/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-half),min(1.,center+half)]

def local_path(value):
    path=Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--reuse-from",type=Path,help="Reuse completed compatible episodes from an earlier evaluation")
    parser.add_argument("--expected-updates",type=int)
    parser.add_argument("--policy-kind",choices=["pretrained","finetuned"])
    parser.add_argument("--language-text")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--horizon",type=int)
    parser.add_argument("--split",choices=["all","pretrain","target","test"])
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()) if args.config else {}
    allowed={"checkpoint","output","expected_updates","seeds","horizon","split","reuse_from","policy_kind","language_text"}
    if not isinstance(config,dict) or set(config)-allowed:
        raise ValueError("Invalid evaluation config")
    checkpoint=local_path(args.checkpoint or config.get("checkpoint",DEFAULT_RUN/"checkpoints/last.ckpt"))
    output=local_path(args.output or config.get("output",DEFAULT_RUN/"robocasa_eval/seed_3_4_5"))
    seeds=args.seeds if args.seeds is not None else config.get("seeds",[3,4,5])
    policy_kind=args.policy_kind or config.get("policy_kind","finetuned")
    updates=args.expected_updates if args.expected_updates is not None else config.get(
        "expected_updates",None if policy_kind=="pretrained" else 10000)
    language_text=args.language_text if args.language_text is not None else config.get("language_text")
    horizon=args.horizon if args.horizon is not None else config.get("horizon",750)
    split=args.split or config.get("split","all")
    reuse_value=args.reuse_from or config.get("reuse_from")
    reuse=local_path(reuse_value) if reuse_value else None
    if (not seeds or any(type(seed) is not int or seed<0 for seed in seeds)
        or len(set(seeds))!=len(seeds) or horizon<1
        or policy_kind not in ["pretrained","finetuned"]
        or (policy_kind=="pretrained" and updates is not None)
        or (policy_kind=="finetuned" and (type(updates) is not int or updates<1))
        or (language_text is not None and (not isinstance(language_text,str)
            or language_text.strip().lower().rstrip(".")!="open the left drawer"))
        or split not in ["all","pretrain","target","test"]):
        raise ValueError("Invalid seeds/update count/horizon/split")
    if not checkpoint.is_file() or not (checkpoint.parent.parent/"normalizer.pkl").is_file():
        raise FileNotFoundError("Checkpoint and companion normalizer are required")
    if reuse is not None:
        source_config=json.loads((reuse/"eval_config.json").read_text())
        if (local_path(source_config["checkpoint"]).resolve()!=checkpoint.resolve()
            or source_config["expected_updates"]!=updates or source_config["horizon"]!=horizon
            or source_config["split"]!=split
            or source_config.get("policy_kind","finetuned")!=policy_kind
            or source_config.get("language_text")!=language_text):
            raise ValueError("Cannot reuse episodes from a different checkpoint or evaluation protocol")
    if output.exists():
        raise FileExistsError(f"Preserving evaluation: {output}")
    output.mkdir(parents=True)
    resolved={"checkpoint":str(checkpoint),"output":str(output),"expected_updates":updates,
              "seeds":seeds,"horizon":horizon,"split":split,"reuse_from":str(reuse) if reuse else None,
              "policy_kind":policy_kind,"language_text":language_text}
    (output/"eval_config.json").write_text(json.dumps(resolved,indent=2)+"\n")
    print("EVAL_START "+json.dumps(resolved),flush=True)
    results=[]
    try:
        for seed in seeds:
            directory=output/f"seed_{seed}"
            source= reuse/f"seed_{seed}" if reuse else None
            reused=source is not None and (source/"summary.json").is_file()
            if reused:
                previous=json.loads((source/"summary.json").read_text())
                if (not previous.get("pipeline_passed") or previous["seed"]!=seed
                    or previous["optimizer_updates"]!=updates
                    or Path(previous["checkpoint"]).resolve()!=checkpoint.resolve()
                    or previous["instruction"].strip().lower().rstrip(".")!="open the left drawer"
                    or previous["actual_control_hz"]!=20 or previous["split"]!=split
                    or any(not (source/name).is_file() for name in
                           ["rollout.mp4","steps.jsonl","protocol.json"])):
                    raise ValueError(f"Incompatible/incomplete cached episode: {source}")
                shutil.copytree(source,directory)
                previous["video"]=str(directory/"rollout.mp4")
                previous["reused_from"]=str(source)
                (directory/"summary.json").write_text(json.dumps(previous,indent=2)+"\n")
                print(f"EPISODE_REUSED seed={seed} source={source}",flush=True)
            else:
                print(f"EPISODE_START seed={seed} checkpoint={checkpoint}",flush=True)
                command=[sys.executable,"-u",str(ROOT/"scripts/evaluate_open_drawer_pretrained_dp.py"),
                         "--checkpoint",str(checkpoint),"--policy-kind",policy_kind,
                         "--seed",str(seed),"--horizon",str(horizon),"--split",split,
                         "--output",str(directory)]
                if updates is not None:
                    command.extend(["--expected-updates",str(updates)])
                if language_text is not None:
                    command.extend(["--language-text",language_text])
                subprocess.run(command,check=True)
            summary=json.loads((directory/"summary.json").read_text())
            keys=["seed","success","steps","reward","clipped_count","replans","wall_clock_s",
                  "inference_s","gpu_peak_allocated_mb","max_drawer_open","action_stats","video"]
            result={key:summary[key] for key in keys}
            metadata=summary.get("env_metadata",{})
            result["layout_id"]=metadata.get("layout_id")
            result["style_id"]=metadata.get("style_id")
            result["reused"]=reused
            results.append(result)
            successes=sum(item["success"] for item in results)
            interval=wilson_interval(successes,len(results))
            print("EPISODE_STATS "+json.dumps(result),flush=True)
            print("dim | min | max | mean | std | p99_abs | clipped",flush=True)
            for action in result["action_stats"]:
                print(f'{action["dim"]:3} | {action["min"]:.4f} | {action["max"]:.4f} | '
                      f'{action["mean"]:.4f} | {action["std"]:.4f} | {action["p99_abs"]:.4f} | '
                      f'{action["clipped_count"]}',flush=True)
            print(f"EVAL_PROGRESS episodes={len(results)}/{len(seeds)} successes={successes} "
                  f"success_rate={successes/len(results):.1%} "
                  f"wilson_95pct=[{interval[0]:.1%},{interval[1]:.1%}]",flush=True)
            (output/"episodes.json").write_text(json.dumps(results,indent=2)+"\n")
        successes=sum(item["success"] for item in results)
        aggregate={**resolved,"expected_finetune_updates":updates,"episodes":len(results),"successes":successes,
                   "success_rate":successes/len(results),
                   "success_rate_wilson_95pct":wilson_interval(successes,len(results)),
                   "episode_stats":results,"reused_episodes":sum(row["reused"] for row in results),
                   "new_episodes":sum(not row["reused"] for row in results),
                   "task":"OpenDrawer","instruction":"open the left drawer",
                   "control_hz":20,"note":"Fixed-seed diagnostic rollouts; interval is descriptive, not a benchmark or power analysis"}
        if policy_kind=="finetuned":
            aggregate["optimizer_updates"]=updates
        (output/"summary.json").write_text(json.dumps(aggregate,indent=2)+"\n")
        print("FINAL_EVAL "+json.dumps(aggregate),flush=True)
        print("seed | success | steps | clips | max_drawer_open | wall_s | layout | style",flush=True)
        for row in results:
            print(f'{row["seed"]:4} | {int(row["success"]):7} | {row["steps"]:5} | '
                  f'{row["clipped_count"]:5} | {row["max_drawer_open"]:.6f} | '
                  f'{row["wall_clock_s"]:.1f} | {row["layout_id"]} | {row["style_id"]}',flush=True)
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps(
            {"completed_episodes":len(results),"requested_seeds":seeds,"error":str(exc)},indent=2)+"\n")
        raise

if __name__=="__main__":
    main()
