# OpenDrawer Pi0.5 three-group exploratory run (2026-10-07)

## Fixed protocol

- All three models initialize independently from the same pinned vanilla pi05_pretrain_human300/75000 checkpoint. No 28 human demonstrations enter these three runs.
- Each group contains exactly eight DreamGen synthetic OpenDrawer episodes. Redundant minimizes mean within-group weighted state/action trajectory distance; diverse maximizes that distance among remaining non-redundant successes; random is seeded sampling from the same success pool. The metric is not KL divergence. Redundant and diverse are disjoint; random may overlap either.
- The selection is in records/opendrawer_synthetic_8group_selection_exploratory_fee2a10f.json. The reviewed labels are in records/opendrawer_dreamgen_motion_annotations.csv: 4 human-confirmed success, 2 partial (excluded), and 21 earlier visual-success candidates. Simulator success and IDM action semantics have not been established. Treat trends as exploratory.
- Every group uses the same three 256x256 camera slots, prompt "open the left drawer", unchanged pretrained z-score normalizer, LoRA on language/action transformers, frozen vision and other base weights, batch 4, 4000 optimizer updates, peak LR 1e-5, seed 0. Save the final 4000-step checkpoint only.
- The DreamGen source is 16 Hz. scripts/prepare_pi05_synthetic_8groups.py converts it to 20 Hz by time-interpolating continuous signals, holding categorical gripper/control mode, and scaling arm per-step OSC_POSE deltas by 16/20. This preserves approximate commanded arm motion per second; it is a modeling assumption shared by all groups. Source images are upscaled 128 to 256 pixels, without gaining detail.
- Evaluate each trained checkpoint on the same 20 RoboCasa seeds used for the vanilla 10/20 comparison, 750 max steps, replan every 5 steps, three-view video for every seed. The script trains and evaluates redundant, then diverse, then random, sequentially.

## Remote Pod

This run was prepared for a fresh RunPod RTX 4090 (24 GB) with 80 GB root disk. The Pod has the pinned OpenPI and RoboCasa environment, kitchen assets, full vanilla checkpoint, the three converted datasets, and the scripts. A read-only preflight successfully decoded all groups through the official Pi0.5 reader on CUDA, including 3 camera masks, 50x12 action windows, and task text. It ran zero training updates and zero rollouts.

The user requested manual ignition. In the Pod's JupyterLab terminal, start once:

    cd /workspace/research/synthetic-utility-smoketest
    nohup bash scripts/run_pi05_synthetic_3groups.sh \
      > /workspace/pi05_synthetic_3groups_9a3dd767.launcher.log 2>&1 \
      < /dev/null &
    echo "launcher PID: $!"

Closing the Windows browser after this does not stop the Pod process. From Mac, log in to the same RunPod account and open this Pod's JupyterLab. Keep the Pod running until results are backed up. The old Pod had no persistent volume; verify storage for this new Pod before stopping it.

Inspect progress:

    cat /workspace/pi05_synthetic_3groups_9a3dd767/current_stage.txt
    tail -n 30 /workspace/pi05_synthetic_3groups_9a3dd767.launcher.log

Results are separated under /workspace/pi05_synthetic_3groups_9a3dd767/{redundant,diverse,random}/: each has train.log, training_summary.json, actual_config.json, rollout.log, rollout/summary.json, rollout/seed_*/rollout.mp4, rollout_results.csv, and result.json. Final summary: /workspace/pi05_synthetic_3groups_9a3dd767/three_group_results.json. Adapter checkpoints are under outputs/OpenDrawer/pi05_synthetic_8_lora/synthetic_<group>_9a3dd767_seed0/4000. ALL_DONE in current_stage.txt means all three rollouts completed; FAILED.txt records a failed stage.
