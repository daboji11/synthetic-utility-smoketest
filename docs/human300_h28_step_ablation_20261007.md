# Human300 + 28 human OpenDrawer LoRA step ablation

All four evaluations used the same 20 OpenDrawer seeds (3, 4, 5, 8, 10, 13,
16, 18, 20, 21, 22, 23, 26, 27, 28, 31, 32, 33, 34, 35). Each LoRA run
started from the same untouched `pi05_pretrain_human300/75000` checkpoint and
used only the 28 left-drawer H50 human demonstrations. The 2000 and 4000
checkpoints came from one 4000-update training run; the older 10000-update
checkpoint came from an earlier run and is reported as a reference, not as a
checkpoint on the same exact optimizer trajectory.

| Policy | Successes / 20 |
| --- | ---: |
| Untouched human300 | 10 |
| human300 + 28 human LoRA, 2000 updates | 9 |
| human300 + 28 human LoRA, 4000 updates | 9 |
| human300 + 28 human LoRA, older 10000 updates | 8 |

The 2000 and 4000 checkpoints differ on four seeds: 2000 succeeds on seeds
4 and 35 where 4000 fails; 4000 succeeds on 18 and 26 where 2000 fails.
Their aggregate result is identical. The 8/20 older run is compatible with
many causes besides update count, including seed-level variance and other
run differences. These 20 rollouts do not establish that excessive steps
caused the decrease. There is no measured improvement over untouched human300
in this OpenDrawer setup.

`records/human300_h28_step_ablation_20seeds.csv` contains all paired outcomes.
The 2000 and 4000 checkpoints, numerical evaluation JSONs, and logs are backed
up under `D:\pi05_human300_h28_4k_backup`. The user is stopping the RunPod;
the remaining rollout videos were intentionally left there and are not backed
up in this folder.
