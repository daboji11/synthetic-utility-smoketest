# ACT branch retained as an engineering record

The user retired ACT as the active baseline on 2026-10-05.
No further ACT tuning or camera identity experiments are scheduled.

Preserved under their original paths:
- outputs/OpenDrawer/real_only/: MimicGen-only debug checkpoints, not a human baseline.
- outputs/OpenDrawer/human_left_only/: H50 left human, single camera, nearest-resampled 16Hz ACT.
- outputs/OpenDrawer/human_left_only_3cam/: H50 left human, three cameras, nearest-resampled 16Hz ACT.
- outputs/OpenDrawer/human_left_only_3cam_camera_id/: separate camera-identity variant if locally trained.
- ACT configs, scripts, custom policy code, and results/robocasa_eval/ diagnostics/videos.

New active pipeline: official LeRobot Diffusion, 28 native-20Hz H50 human-left demonstrations,
three 128x128 cameras, 16D state and 12D action; RoboCasa control at 20Hz.
No ACT weights, ACT-specific processors, or ACT horizon settings are used.
