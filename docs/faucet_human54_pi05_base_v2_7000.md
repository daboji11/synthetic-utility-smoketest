# Faucet human54: fresh-base 7000-step diagnostic run

Status: scripts prepared and CPU checks performed; GPU execution remains manual.

This tests whether human54 LoRA begins to learn after correcting state token
saturation and replacing the old 3000-step learning-rate schedule. It changes
both normalization and LR, so any improvement cannot be attributed to LR alone.
The prior 0/20, 1/20, 0/20 results did not use verified identical scenes.

## Fixed protocol

| Setting | Value |
|---|---|
| Task / prompt | `TurnOnSinkFaucet` / `turn on the sink faucet` |
| Initialization | Original official `pi05_base`, resolved from the pinned existing config |
| OpenPI revision | `5a6beda9ff99da30b4e1b59320f6a32971d7c397` |
| Data | `data/processed/TurnOnSinkFaucet/h50_human_native_dp_20fps_256` |
| Full dataset | 54 original H50 humans, 12070 frames, 162 camera videos, 20 FPS |
| Training | LoRA only; vision and action/time projection heads frozen; fresh optimizer |
| Batch / seed | 8 / 0; single RTX 5090 32 GB |
| LR | 200 warmup steps; cosine peak `1e-5`, decay through step 7000 to `1e-6` |
| Saves | 1000, 3000, 5000, 7000; complete inference params and matching assets |
| EMA | None |
| State / action | 16 / 12 dimensions, padded to 32; native ordering |
| Camera mapping | base = left external; left wrist = wrist; right wrist slot = right external |
| Horizon | Predict 50 actions; execute 5 per replan; episode cap 750 at 20 Hz |
| Evaluation | 20 seed IDs: 3,4,5,8,10,13,16,18,20,21,22,23,26,27,28,31,32,33,34,35 |

`pi05_base` is a pretrained base model, not an untrained model. Both the probe
and the full run load the original base again. Probe weights are never reused
for the full run. No human300 or previous adapter weights are loaded.

## State normalization

The old z-score position inputs put 54.67%, 48.14%, 42.46% of x/y/z values
in the endpoint bins of the pinned `[-1,1]` state tokenizer. The new mapping
uses center `(q01+q99)/2` and scale `(q99-q01)/2` for state only. On all 12070
local human frames the endpoint fractions become 2.212%, 2.104%, 2.171%.
This removes a confirmed information loss; it does not guarantee success.

The upstream affine normalizer still uses `use_quantile_norm=False` globally.
State asset fields `mean` and `std` contain these affine centers/scales, **not
empirical means/stds**. Scales below `1e-6` use 1 to avoid amplifying numerical
noise in constant dimensions. Actions retain their exact original empirical
mean/std and upstream epsilon, so action units and inverse normalization stay
the same. Padding retains center 0 / scale 1. Training and policy loading use
the same saved assets, and evaluation checks those against the v2 recipe.
The old base interface assets and old runs remain separate.

## Automatic small learning check

`all` first runs a 300-update probe on two original human episodes selected by
the pinned Groot `2_demos` filter with selection seed 0. It retains full-human54
normalization and the same 7000-step schedule (warmup 200). Selected IDs,
fixed observation batch, metrics and before/after predictions are recorded.

For a fixed batch of eight training observations, it compares sampled actions
using noise seeds 101/102/103 before and after the probe. Reports contain raw
arm RMSE per dimension for the first 1/5/50 actions. The gate requires at least
10% reduction in normalized MSE over the first 5 executed actions and first
6 arm dimensions. This is a practical learning-response gate, **not proof of
accurate demo reproduction, held-out generalization, or closed-loop success**.
The 10% threshold is a chosen diagnostic cutoff, not a published criterion.

If this check fails or metrics become nonfinite, the script stops before the
full run. The instance remains on: inspect the logs and shut it down manually
if further GPU work is not needed. The probe adds an inference/training compile
and a fresh full-run initialization; first output can take a few minutes.

## Deploy and launch tomorrow

The cloud environment and weights already prepared should be reused. If it is a
Git checkout, update it using `git pull --ff-only`. If the uploaded cloud folder
has no `.git`, upload `faucet_v2_7000_code.tar.gz` from the local prepared bundle
and extract it into that folder; the bundle contains code/docs/tests only.

After the updated files are present, paste this **once** in the AutoDL terminal:

```bash
cd /root/autodl-tmp/research/synthetic-utility-smoketest
mkdir -p logs/pi05_base_faucet_v2/human54_lr1e5_7000_seed0
nohup bash scripts/run_pi05_base_faucet_v2_5090.sh all \
  > logs/pi05_base_faucet_v2/human54_lr1e5_7000_seed0/launcher.log 2>&1 < /dev/null &
echo "Launcher PID: $!"
```

This runs preflight → two-demo probe → **fresh-base** human54 7000-step training
→ final-checkpoint 20-rollout evaluation. It does not rerun the old base 0/20
baseline. The four checkpoint saves do not trigger four evaluations: only
7000 is evaluated automatically. The other checkpoints are available for
separate diagnosis.

To see explicit training and rollout output:

```bash
cd /root/autodl-tmp/research/synthetic-utility-smoketest
tail -f logs/pi05_base_faucet_v2/human54_lr1e5_7000_seed0/launcher.log
```

`Ctrl+C` exits the log viewer; the background launcher continues. You can close
the laptop/browser and reconnect later while the cloud instance stays running.
For a snapshot:

```bash
bash scripts/run_pi05_base_faucet_v2_5090.sh status
```

Preflight requires at least **30 GB free** for the four full checkpoints and
checks the source, base manifest, actual human arrays and normalization. Existing
output directories are preserved; rerunning `all` with the same run ID fails
instead of overwriting. If the probe passed and the full run has not started,
use the `train` stage separately. For a genuinely new run, set a new
`FAUCET_V2_RUN_ID` consistently for every stage. A shared lock prevents concurrent
v2 stages. Avoid running another old training/evaluation job on the GPU.

## Outputs and additional evaluation

Full checkpoints and training records:

```text
outputs/TurnOnSinkFaucet/pi05_base_faucet_state_range_v2/human54_lr1e5_7000_seed0/
```

Probe records:

```text
outputs/TurnOnSinkFaucet/pi05_base_faucet_state_range_v2_probe/human54_lr1e5_7000_seed0/
```

Rollouts, per-episode outcomes, three-camera videos and initial-state records:

```text
results/robocasa_eval/pi05_base_faucet_v2/human54_lr1e5_7000_seed0/eval-7000/
```

Each episode also saves initial XML, simulator state and episode metadata for
diagnosis. Exact portable scene replay remains unvalidated; results explicitly
record `paired_initial_conditions=false`. Same seed IDs alone do not establish
identical scene conditions, so comparisons with old runs are exploratory.

To evaluate an earlier save manually, for example:

```bash
bash scripts/run_pi05_base_faucet_v2_5090.sh eval-3000
```

Logs include loss, LR, gradient norm and LoRA update norm. Loss is upstream
flow-matching velocity MSE averaged over 32 dimensions and the full chunk;
`0.1` is not itself a success/failure cutoff. Use generated action error and
rollout behavior alongside it. No synthetic experiment should follow until
real-demo adaptation is demonstrated.

## Validation limits

Local CPU checks cover affine state ordering, constant dimensions, action
normalization roundtrip using the actual pinned upstream transform, saved-asset
verification, LR values, frozen vision/heads, probe rejection, manual launch
requirement and launcher error propagation/order. A CPU scan verifies all
12070 real state/action rows and stored statistics. Python compilation and Bash
syntax are checked. GPU compilation, the full model probe and real 20-rollout
execution still require the user's manual cloud launch.
