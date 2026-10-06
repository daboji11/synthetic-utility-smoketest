# Pi0.5 synthetic eight-episode handoff (2026-10-06)

Decision: Tomorrow, train three LoRA models starting separately from the same
pinned RoboCasa Pi0.5 vanilla checkpoint. Each gets only eight synthetic
OpenDrawer episodes; do not include the 28 H50 human episodes. Groups are
redundant, diverse, and random. User explicitly asked to stop now and resume
locally tomorrow morning, then push completed small files to Git before school.
No training or Git push was started today.

## Saved work

- scripts/select_opendrawer_synthetic_groups.py computes three deterministic
  groups from one success-screened pool. The metric is normalized weighted
  state/action trajectory distance, NOT KL divergence.
- records/opendrawer_synthetic_8group_selection_draft.json and .csv:
  redundant: 420 421 422 423 432 470 471 474; mean pairwise 0.73862.
  diverse: 431 437 439 455 463 466 476 479; mean pairwise 1.21184.
  random: 422 423 430 433 437 439 455 470; mean pairwise 1.06323.
  These are PROVISIONAL and must not be used as final training IDs.
- scripts/audit_opendrawer_synthetic_source.py and
  records/opendrawer_synthetic_source_audit_draft.json check 18 unique draft
  episodes: finite 12D actions, 16D states, 279 frames at 16 Hz, timestamps,
  and three-camera preview availability. The audit does not prove simulator
  success or IDM action validity.
- records/opendrawer_dreamgen_motion_annotations.csv is live annotation output;
  preserve the user's changes. At this handoff 22 candidates are provisional
  successes, only episode 439 has a human success override, and no episode has
  motion_reviewed=true. Classifier is at http://localhost:8765/.
- Prior all-60 trajectory rankings cannot be reused directly. Their
  redundant_10 has no provisionally successful videos.
- Previous fine-tune audit: docs/finetune_diagnosis_20261006.md and
  records/pi05_frozen_weights_audit.json.

## Tomorrow

1. Read latest annotations. Final group selection requires at least 16
   manually confirmed successes also marked motion reviewed. If not enough,
   continue review; never fill with failed or uncertain videos.
2. Run:
   python3 scripts/select_opendrawer_synthetic_groups.py --mode final
   Then audit the resulting final JSON with
   scripts/audit_opendrawer_synthetic_source.py in synutility-dp-pretrained.
3. Build three separately pinned datasets. Source DreamGen is 16 Hz versus
   prior human training and rollout at 20 Hz. Choose and verify temporal
   conversion on one episode before all groups. The RoboCasa arm controller
   has OSC_POSE input_type=delta, so simply duplicating commands at 20 Hz
   would change action speed. Gripper/control fields are categorical.
4. Parameterize the Pi0.5 trainer to accept an eight-episode synthetic dataset;
   the present script hardcodes 28 H50 human episodes. Hold checkpoint,
   normalization, adapter scope, batch, LR, update count, preprocessing,
   training seed and rollout seeds fixed across all groups. Each run must
   load vanilla directly, not a previous group's adapter. Fix settings before
   inspecting success results.
5. Train and evaluate only on the remote 4090 after the user resumes.
   Compare paired per-seed outcomes across the three groups and retain vanilla
   as a reference. Treat any trend as exploratory; replicate on the next task.
6. Commit and push scripts, docs and small CSV/JSON results tomorrow as the
   user requested. Exclude cached data, videos, checkpoints and transient files.
   This handoff is currently local, not pushed.

Pinned source revision:
686cd91d4502b0606e473b07adb73c920b05d4e3
