"""Read-only three-camera ACT audit: data quality, forward trace, and input sensitivity."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data._utils.collate import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/robocasa_eval/camera_audit_3cam"
CAMERAS = [
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
]


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserving existing audit: {OUT}")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "evaluation", ROOT / "scripts/evaluate_open_drawer_h50_act_robocasa.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    evaluation.DATASET = ROOT / "data/processed/OpenDrawer/h50_human_left_16fps_3cam"
    evaluation.CHECKPOINT = ROOT / "outputs/OpenDrawer/human_left_only_3cam/seed_0/checkpoints/last/pretrained_model"
    evaluation.REPO_ID = "local/open_drawer_h50_human_left_16fps_3cam"
    evaluation.CAMERAS = CAMERAS
    features, policy, preprocessor, postprocessor = evaluation.make_policy(
        device="cpu", require_training_step=10000)
    manifest = json.loads((evaluation.DATASET / "resampling_manifest.json").read_text())
    dataset = LeRobotDataset(evaluation.REPO_ID, root=evaluation.DATASET,
                             video_backend="torchcodec",
                             delta_timestamps={"action": [i / 16 for i in range(100)]})
    lengths = [item["target_length"] for item in manifest["episodes"]]
    offsets = np.cumsum([0, *lengths[:-1]]).tolist()
    images_by_camera = {key: [] for key in CAMERAS}
    pairs = {(0, 1): [], (0, 2): [], (1, 2): []}
    for ep, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        for local_index in sorted({0, length // 2, length - 1}):
            frame = dataset[offset + local_index]
            for key in CAMERAS:
                image = frame[key].numpy()
                images_by_camera[key].append({
                    "episode": ep, "frame": local_index, "mean": float(image.mean()),
                    "std": float(image.std()), "all_black": bool(np.all(image == 0)),
                    "finite": bool(np.isfinite(image).all()),
                })
            for left, right in pairs:
                pairs[left, right].append(float(
                    (frame[CAMERAS[left]] - frame[CAMERAS[right]]).abs().mean()))
    quality = {
        key: {"sampled_frames": len(rows), "black_frames": sum(row["all_black"] for row in rows),
              "nonfinite_frames": sum(not row["finite"] for row in rows),
              "min_pixel_std": min(row["std"] for row in rows),
              "mean_pixel_std": float(np.mean([row["std"] for row in rows]))}
        for key, rows in images_by_camera.items()
    }
    duplicate_views = {
        f"{CAMERAS[left]} vs {CAMERAS[right]}": {
            "identical_pairs": sum(value == 0 for value in differences),
            "sampled_pairs": len(differences), "mean_pixel_abs_difference": float(np.mean(differences)),
        }
        for (left, right), differences in pairs.items()
    }
    print("DATA_QUALITY", json.dumps(quality), flush=True)
    indices = [offsets[0] + lengths[0] // 4, offsets[5] + lengths[5] // 2,
               offsets[12] + lengths[12] // 2, offsets[21] + lengths[21] // 3]
    raw_batch = default_collate([dataset[index] for index in indices])
    prepared = preprocessor(dict(raw_batch))
    backbone_calls, encoder_calls = [], []
    def trace_backbone(module, inputs, output):
        img = inputs[0]
        matched = [key for key in CAMERAS if img is prepared[key]]
        backbone_calls.append({
            "camera": matched, "input_shape": list(img.shape),
            "feature_map_shape": list(output["feature_map"].shape),
        })
    def trace_encoder(module, inputs):
        encoder_calls.append(list(inputs[0].shape))
    handles = [
        policy.model.backbone.register_forward_hook(trace_backbone),
        policy.model.encoder.register_forward_pre_hook(trace_encoder),
    ]
    try:
        # The same policy.forward entry used by training, without gradients or optimizer updates.
        with torch.inference_mode():
            loss, loss_info = policy.forward(prepared)
    finally:
        for handle in handles:
            handle.remove()
    if [call["camera"] for call in backbone_calls] != [[key] for key in CAMERAS]:
        raise ValueError(f"Unexpected camera consumption: {backbone_calls}")
    print("FORWARD_TRACE", json.dumps({
        "backbone_calls": backbone_calls, "encoder_shapes": encoder_calls,
        "finite_loss": bool(torch.isfinite(loss)), "optimizer_steps": 0}), flush=True)

    def infer(batch):
        prepared_batch = preprocessor(dict(batch))
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(prepared_batch)
            return postprocessor(chunk).cpu().numpy()
    baseline = infer(raw_batch)
    sensitivity = {}
    for key in CAMERAS:
        for perturbation in ("black_image", "image_from_other_demo"):
            changed = dict(raw_batch)
            changed[key] = (torch.zeros_like(raw_batch[key]) if perturbation == "black_image"
                            else torch.roll(raw_batch[key], 1, dims=0))
            delta = infer(changed) - baseline
            sensitivity[f"{key}:{perturbation}"] = {
                "mean_abs_raw_action_change": float(np.abs(delta).mean()),
                "arm_mean_abs_change": float(np.abs(delta[:, :, :6]).mean()),
                "gripper_mean_abs_change": float(np.abs(delta[:, :, 6]).mean()),
                "max_abs_change": float(np.abs(delta).max()),
                "first_action_arm_mean_abs_change": float(np.abs(delta[:, 0, :6]).mean()),
            }
    changed = dict(raw_batch)
    changed["observation.state"] = torch.roll(raw_batch["observation.state"], 1, dims=0)
    delta = infer(changed) - baseline
    sensitivity["state_from_other_demo"] = {
        "arm_mean_abs_change": float(np.abs(delta[:, :, :6]).mean()),
        "gripper_mean_abs_change": float(np.abs(delta[:, :, 6]).mean()),
    }
    changed = dict(raw_batch)
    changed[CAMERAS[0]], changed[CAMERAS[1]] = raw_batch[CAMERAS[1]], raw_batch[CAMERAS[0]]
    delta = infer(changed) - baseline
    permutation = {
        "swapped_cameras": CAMERAS[:2],
        "mean_abs_raw_action_change": float(np.abs(delta).mean()),
        "max_abs_change": float(np.abs(delta).max()),
        "interpretation": "Sensitivity probe only; camera use does not establish useful spatial learning.",
    }
    targets = raw_batch["action"].numpy()
    valid = ~raw_batch["action_is_pad"].numpy()
    mae = [float(np.abs(targets[:, :, dim] - baseline[:, :, dim])[valid].mean())
           for dim in range(12)]
    old_checkpoint = ROOT / "outputs/OpenDrawer/human_left_only/seed_0/checkpoints/last/pretrained_model"
    with safe_open(evaluation.CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as new_reader:
        with safe_open(old_checkpoint / "model.safetensors", framework="pt", device="cpu") as old_reader:
            new_shapes = {key: new_reader.get_slice(key).get_shape() for key in new_reader.keys()}
            old_shapes = {key: old_reader.get_slice(key).get_shape() for key in old_reader.keys()}
            examples = {}
            for key in ("model.backbone.conv1.weight", "model.action_head.weight",
                        "model.encoder_robot_state_input_proj.weight"):
                if key in new_shapes and key in old_shapes:
                    delta_weights = new_reader.get_tensor(key) - old_reader.get_tensor(key)
                    examples[key] = {
                        "identical": bool(torch.equal(new_reader.get_tensor(key), old_reader.get_tensor(key))),
                        "mean_abs_difference": float(delta_weights.abs().mean()),
                    }
    train_lib = Path("/home/dabozi/miniconda3/envs/synutility312/lib/python3.12/site-packages/lerobot")
    sim_lib = Path("/home/dabozi/miniconda3/envs/synutility-sim/lib/python3.12/site-packages/lerobot")
    source_parity = {}
    for relative in ("policies/act/modeling_act.py", "policies/act/configuration_act.py",
                     "utils/feature_utils.py", "datasets/lerobot_dataset.py"):
        train_hash = hashlib.sha256((train_lib / relative).read_bytes()).hexdigest()
        sim_hash = hashlib.sha256((sim_lib / relative).read_bytes()).hexdigest()
        source_parity[relative] = {"identical": train_hash == sim_hash,
                                   "training_sha256": train_hash, "evaluation_sha256": sim_hash}
    OUT.mkdir(parents=True)
    # Contact sheet via the already-installed video library; no extra dependencies.
    canvas = np.full((3 * 152, 3 * 128, 3), 245, dtype=np.uint8)
    for col, key in enumerate(CAMERAS):
        image = np.rint(raw_batch[key][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas[24:152, col*128:(col+1)*128] = image
        cv2.putText(canvas, ["Train: left", "Train: right", "Train: wrist"][col],
                    (col*128+3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    video_path = ROOT / "outputs/OpenDrawer/human_left_only_3cam/seed_0/robocasa_eval/three_camera_seed_3_4_5/videos/seed_3.mp4"
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot read saved evaluation video: {video_path}")
    for row, frame_index in enumerate((0, 100), start=1):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(f"Missing video frame {frame_index}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas[row*152+24:(row+1)*152] = rgb[:128]
        cv2.putText(canvas, f"Rollout seed=3 step={frame_index}", (3,row*152+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    video.release()
    image_path = OUT / "training_vs_rollout_views.png"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Contact sheet write failed")
    report = {
        "checkpoint": str(evaluation.CHECKPOINT), "device": "cpu",
        "training_step": policy._eval_validation["training_step"],
        "camera_keys": CAMERAS, "model_parameters": sum(p.numel() for p in policy.parameters()),
        "camera_quality": quality, "duplicate_view_check": duplicate_views,
        "sample_indices": indices, "backbone_forward_calls": backbone_calls,
        "transformer_encoder_input_shapes": encoder_calls,
        "same_parameter_shapes_as_single_camera": new_shapes == old_shapes,
        "representative_weight_comparison": examples, "training_eval_source_parity": source_parity,
        "camera_sensitivity": sensitivity, "camera_swap_probe": permutation,
        "training_sample_raw_action_mae_by_dim": mae,
        "probe_notes": "4 training observations, deterministic eval-mode forward; not a held-out performance estimate. Perturbations measure sensitivity, not causal usefulness.",
        "optimizer_steps": 0, "simulator_rollouts": 0,
        "view_contact_sheet": str(image_path),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CAMERA_SENSITIVITY", json.dumps(sensitivity), flush=True)
    print("CAMERA_SWAP", json.dumps(permutation), flush=True)
    print("AUDIT_REPORT", str(OUT / "report.json"), flush=True)


if __name__ == "__main__":
    main()
