#!/usr/bin/env python
"""Run fixed seeds3/4/5 sequentially, preserving explicit stats and videos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"outputs/OpenDrawer/dp_pretrained_human_left/seed_0"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=OUTPUT/"checkpoints/last.ckpt")
    parser.add_argument("--output",type=Path,default=OUTPUT/"robocasa_eval/seed_3_4_5")
    parser.add_argument("--expected-updates",type=int,default=10000)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving evaluation: {args.output}")
    args.output.mkdir(parents=True)
    results=[]
    for seed in [3,4,5]:
        directory=args.output/f"seed_{seed}"
        print(f"EPISODE_START seed={seed} checkpoint={args.checkpoint}",flush=True)
        subprocess.run([sys.executable,"-u",str(ROOT/"scripts/evaluate_open_drawer_pretrained_dp.py"),
                        "--checkpoint",str(args.checkpoint),"--expected-updates",str(args.expected_updates),
                        "--seed",str(seed),"--output",str(directory)],check=True)
        summary=json.loads((directory/"summary.json").read_text())
        result={key:summary[key] for key in ["seed","success","steps","reward","clipped_count","replans","wall_clock_s","video"]}
        results.append(result)
        successes=sum(item["success"] for item in results)
        print("EPISODE_STATS "+json.dumps(result),flush=True)
        print(f"EVAL_PROGRESS episodes={len(results)}/3 successes={successes} success_rate={successes/len(results):.1%}",flush=True)
        (args.output/"episodes.json").write_text(json.dumps(results,indent=2)+"\n")
    summary={"checkpoint":str(args.checkpoint),"optimizer_updates":args.expected_updates,"seeds":[3,4,5],
             "episodes":3,"successes":sum(item["success"] for item in results),
             "success_rate":sum(item["success"] for item in results)/3,
             "episode_stats":results,"task":"OpenDrawer","instruction":"open the left drawer",
             "control_hz":20,"note":"Three-episode initial check, not a precise benchmark estimate"}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("FINAL_EVAL "+json.dumps(summary),flush=True)
if __name__=="__main__":
    main()
