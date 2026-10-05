# Local inference of the completed4090 H50 DP
The downloaded archive is extracted into outputs/OpenDrawer/dp_pretrained_human_left/seed_0_4090_archive.
This is the completed batch4 /10000-update checkpoint. New batch32 training settings do not change it.

The evaluation config is configs/native_dp/h50_left_local_eval.yaml.
OpenDrawer left drawer only, seeds3,4,5,8,10, split=all,20Hz,750-step horizon,
three256x256 cameras, official EMA,2 observations,8 actions per inference,100 DDPM denoising steps.
Seeds were verified by environment reset;6,7,9 are right drawer and are excluded.
No policy inference or environment action was executed while preparing this run.

From a local WSL Ubuntu terminal:

    cd /home/dabozi/research/synthetic-utility-smoketest
    source /home/dabozi/miniconda3/etc/profile.d/conda.sh
    conda activate synutility-dp-pretrained
    bash scripts/run_h50_native_dp_local_eval.sh

Each invocation creates a new timestamped directory under results/robocasa_eval and a matching console log.
It prints per-episode success/reward, per-dimension action stats, drawer opening, GPU memory,
wall-clock time and aggregate success rate with a Wilson95% descriptive interval.
Each seed saves a three-camera video, steps.jsonl, protocol.json and summary.json.
The aggregate saves eval_config.json, episodes.json and summary.json.

Five fixed seeds are an initial diagnostic sample, not a precise population success-rate or statistical-power estimate.
The config and result protocol record the exact evaluated checkpoint and task/control convention.
Inference batch size is1; training batch size is recorded separately.
Default float32 architecture is unchanged. Full inference peak memory is only measured when the user launches it.

## Extend the existing5 episodes to20

Wait for the running five-episode evaluation to finish, then from the same activated WSL terminal:

    bash scripts/run_h50_native_dp_local_eval_20.sh

Config: configs/native_dp/h50_left_local_eval_20.yaml.
Verified left seeds:3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35.
The checkpoint, camera, control frequency and action-chunk protocol are identical.
The completed episodes in the initial5-seed directory are validated and copied into the new evaluation.
A different checkpoint/horizon/split is rejected; incomplete episodes are run anew in the new directory.
The old evaluation files are preserved. Console prints EPISODE_REUSED and counts reused/new episodes.
Fresh output: results/robocasa_eval/h50_dp_local_20_seeds_<timestamp>, plus a matching.log.
The final statistic counts20 unique seeds, including the reused results.

To run all20 fresh instead, override the reuse directory only through the config or use the seed evaluator
without reuse_from in a separate config; do not silently combine a different policy's results.

## Original pretrained control, no H50 fine-tuning

Start with one paired seed:

    bash scripts/run_pretrained_dp_local_eval_20.sh --seeds 3

Run the full20-seed control:

    bash scripts/run_pretrained_dp_local_eval_20.sh

This uses the original published pretrain_human300 EMA and companion normalizer in the HF snapshot.
It enforces checkpoint task=pretrain_human300 and does not apply the H50 optimizer-update gate.
All20 seeds are run fresh; fine-tuned results cannot be reused.
The seeds/environment/cameras/control/chunk/horizon are identical to the H50 fine-tuned20-seed test.
The literal policy language is explicitly open the left drawer for both conditions.
Weights/normalizer remain unmodified; this text override is recorded in each protocol.
Fresh directory: results/robocasa_eval/pretrained_dp_local_20_seeds_<timestamp>.
The original single-episode evaluator still defaults to the native environment instruction when no text override is passed.

If the fine-tuned evaluation is still active, finish or interrupt it before launching the control.
Completed fine-tuned episode outputs remain available; interrupting may leave the current episode incomplete.
Read success/reward, drawer motion and per-seed videos before attributing failure to fine-tuning or the interface.
