# OpenDrawer synthetic motion annotation

Manual annotation for the 60 cached DreamGen previews, with synchronized left/right/wrist views. Uses no GPU; does not start training, infer actions, select training samples, or modify the prior labels.

## Build the three-camera previews

Run scripts/build_open_drawer_three_camera_previews.py in the existing
synutility-dp-pretrained environment (requires pyarrow and ffmpeg).
It uses the pinned source revision and per-camera episode timestamps.
The three-camera manifest records paths, frame counts, layout and checksums.
All three views are encoded into one video, so playback and seeking stay synchronized.

## Start in WSL

```bash
cd /home/dabozi/research/synthetic-utility-smoketest
python3 scripts/serve_open_drawer_annotations.py --port 8765
```

Open http://localhost:8765/ in the Windows browser. Default: 21 provisional success videos; 33 failure and 6 uncertain also available. These are prior assistant visual assessments, not simulator labels or human ground truth.

## Annotate

- Choose approach side/angle, contact position/quality, approach shape, pull direction/style, final opening amount.
- Use unclear where the three 128x128 camera views is insufficient. Angles are coarse projected categories, not measured 3D angles.
- A–E are your manual groups; explain the grouping rules in notes.
- Optional manual_visual_label records your outcome reassessment separately from the preserved visual_label.
- Check motion reviewed when finished with an episode.
- Edits autosave to records/opendrawer_dreamgen_motion_annotations.csv. Wait for saved status before closing; export CSV for a portable copy.
- Import old open_drawer_annotations.csv by episode_id. Browser restoration requires the same browser origin as the old tool.
- All 60 rows and source provenance are retained; unknown fields stay blank. No automatic simulator_success, official_success_label, or finetune_selected.

Original records/opendrawer_dreamgen_video_labels.csv remains unchanged. Server binds to localhost. Stale concurrent writes are rejected: export your draft, refresh, and import it to merge.

Prior success labels were assessed from the left-camera previews; those original labels remain unchanged. Three-camera human reassessments belong in manual_visual_label.
