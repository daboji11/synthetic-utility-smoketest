> Historical 3000-step protocol. For the new independent 7000-step diagnostic run, see [v2 instructions](faucet_human54_pi05_base_v2_7000.md).

# TurnOnSinkFaucet: pi05_base + 54 human H50 demos

This is the human-only validation before any synthetic selection experiment.
Training and evaluation use the same pinned RoboCasa/OpenPI interface as the
OpenDrawer study, with the task changed to `TurnOnSinkFaucet`.

## Fixed protocol

- **Initialization:** official untouched `gs://openpi-assets/checkpoints/pi05_base/params` from the pinned RoboCasa OpenPI config (`5a6beda9ff99da30b4e1b59320f6a32971d7c397`). No human300 or prior LoRA weights.
- **Data:** exactly H50 human episodes 1132–1185, 54 episodes and 12,070 frames. All carry `turn on the sink faucet`; no synthetic episodes.
- **Observations:** 3 cameras, 256×256, 20 Hz; 16D state reordered by the pinned RoboCasa interface.
- **Actions:** original 12D H50 values, 50-step action horizon, 20 Hz control. No action resampling. Gripper and control mode are fixed at −1 in this source task.
- **Normalization:** mean/std from the 54 H50 human episodes. The same assets are used for untouched-base and LoRA evaluation; base weights remain unchanged.
- **LoRA:** both vision-language backbone and action expert use the pinned LoRA variants; only parameters with `lora` in their path train. Batch 8, 3000 optimizer steps, peak LR `5e-6`, seed 0, four loader workers. Checkpoints at 500, 1000, 2000, 3000.
- **Evaluation:** 20 fixed seeds `(3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35)`, 750-step maximum, replan every 5 environment steps. The evaluator stores each initial state's vector and three image hashes and requires exact matching in LoRA runs.

## On AutoDL

Project: `/root/autodl-tmp/research/synthetic-utility-smoketest`.
All large caches and outputs are on the 120 GB data disk.

The user starts each measured stage manually in the Jupyter terminal, where
the log is printed live and also saved under `logs/pi05_base_faucet/`:

```bash
cd /root/autodl-tmp/research/synthetic-utility-smoketest
bash scripts/run_pi05_base_faucet_5090.sh preflight
bash scripts/run_pi05_base_faucet_5090.sh baseline
bash scripts/run_pi05_base_faucet_5090.sh train
bash scripts/run_pi05_base_faucet_5090.sh eval-3000
bash scripts/run_pi05_base_faucet_5090.sh compare
```

If the 3000-step checkpoint underperforms, evaluate `eval-500`, `eval-1000`,
or `eval-2000` on the same matched initial states. Do not infer improvement
from training loss alone. Synthetic groups follow only after this human check.
