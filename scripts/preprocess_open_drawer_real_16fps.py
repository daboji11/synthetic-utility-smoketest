"""Build an immutable, single-camera OpenDrawer real dataset at 16 fps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchcodec.decoders import VideoDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets import RoboCasaLeRobotDataset

CAMERA = "observation.images.robot0_agentview_left_image"
REPO_ID = "local/open_drawer_real_16fps"
RAW_CACHE = ROOT / "data/raw/huggingface"
DESTINATION = ROOT / "data/processed/OpenDrawer/real_16fps"
STAGING = DESTINATION.with_name(DESTINATION.name + ".incomplete")


def selected_indices(length: int) -> list[int]:
    """Nearest 20 Hz source frame for j/16; exact ties select the earlier frame."""
    target_count = (4 * length + 4) // 5
    indices = [min((5 * j + 1) // 4, length - 1) for j in range(target_count)]
    if not indices or indices[-1] >= length or len(set(indices)) != len(indices):
        raise ValueError(f"Invalid resampling map for source length {length}")
    return indices


def verify_and_publish() -> None:
    manifest = json.loads((STAGING / "resampling_manifest.json").read_text())
    verified = LeRobotDataset(REPO_ID, root=STAGING, video_backend="torchcodec")
    if verified.num_episodes != 21 or verified.meta.fps != 16:
        raise ValueError("Output episode count or FPS mismatch")
    if verified.meta.camera_keys != [CAMERA]:
        raise ValueError("Output camera keys mismatch")
    if tuple(verified.features["action"]["shape"]) != (12,) or tuple(
        verified.features["observation.state"]["shape"]
    ) != (16,):
        raise ValueError("Output action/state feature mismatch")
    expected_frames = sum(item["target_length"] for item in manifest["episodes"])
    if len(manifest["episodes"]) != 21 or len(verified) != expected_frames:
        raise ValueError("Output frame or manifest count mismatch")
    offset = 0
    for item in manifest["episodes"]:
        episode_id, length = item["new_episode_index"], item["target_length"]
        for local_index in (0, length - 1):
            frame = verified[offset + local_index]
            if (int(frame["episode_index"]) != episode_id
                or abs(float(frame["timestamp"]) - local_index / 16) > 1e-4
                or tuple(frame[CAMERA].shape) != (3, 128, 128)
                or tuple(frame["action"].shape) != (12,)
                or tuple(frame["observation.state"].shape) != (16,)):
                raise ValueError(f"Output frame mismatch in episode {episode_id}")
        offset += length
    STAGING.rename(DESTINATION)
    print(f"verified: episodes=21 fps=16 frames={expected_frames} "
          f"action_dim=12 state_dim=16 camera={CAMERA}", flush=True)
    print(f"dataset: {DESTINATION}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Verify and publish an already completed staging dataset")
    args = parser.parse_args()
    if args.finalize_existing:
        if DESTINATION.exists() or not STAGING.is_dir():
            raise FileNotFoundError("Expected only the completed staging dataset")
        verify_and_publish()
        return
    if DESTINATION.exists() or STAGING.exists():
        raise FileExistsError("Output already exists; preserve the previous dataset")
    spec = yaml.safe_load((ROOT / "configs/dataset/open_drawer.yaml").read_text())
    source = RoboCasaLeRobotDataset(
        **spec["real"], task_id="OpenDrawer", camera_key=CAMERA,
        cache_dir=RAW_CACHE, offline=True,
    )
    if len(source) != 21 or source.info["fps"] != 20 or source.camera_key != CAMERA:
        raise ValueError("Unexpected source dataset or camera")
    features = {
        "action": source.features["action"].copy(),
        "observation.state": source.features["observation.state"].copy(),
        CAMERA: {"dtype": "video", "shape": [128, 128, 3],
                 "names": ["height", "width", "channel"]},
    }
    first = source.get_episode_metadata(source.get_episode_ids()[0])
    data_path = source.info["data_path"].format(
        chunk_index=first["data/chunk_index"], file_index=first["data/file_index"]
    )
    timestamps = pq.read_table(source._download(data_path), columns=["timestamp"]).column(
        "timestamp"
    ).to_numpy()
    writer = LeRobotDataset.create(
        repo_id=REPO_ID, root=STAGING, fps=16, features=features,
        robot_type=source.info.get("robot_type"), use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p",
                                     g=16, crf=18, preset="veryfast"),
    )
    video_decoders: dict[Path, VideoDecoder] = {}
    manifest = {"source_repo": source.repo_id, "source_revision": source.revision,
                "source_fps": 20, "target_fps": 16,
                "resampling": "nearest timestamp j/16, ties to earlier source frame",
                "camera_key": CAMERA, "episodes": []}
    for new_episode_index, source_id in enumerate(source.get_episode_ids()):
        episode = source.get_episode_metadata(source_id)
        length = int(episode["length"])
        start = int(episode["dataset_from_index"])
        indices = selected_indices(length)
        source_times = timestamps[start : start + length]
        if len(source_times) != length or not np.allclose(
            source_times, np.arange(length) / 20, rtol=0, atol=1e-4
        ):
            raise ValueError(f"Unexpected source timestamps for episode {source_id}")
        video, video_start, _ = source.video_segment(source_id)
        if abs(video_start - start / 20) > 1e-3:
            raise ValueError(f"Video and parquet start disagree for episode {source_id}")
        if video not in video_decoders:
            video_decoders[video] = VideoDecoder(str(video))
        decoder = video_decoders[video]
        frames = decoder.get_frames_at([start + index for index in indices])
        if not np.allclose(frames.pts_seconds.numpy(),
                           (start + np.asarray(indices)) / 20, rtol=0, atol=1e-3):
            raise ValueError(f"Video frames misaligned for episode {source_id}")
        images = frames.data.permute(0, 2, 3, 1).numpy()
        trajectory = source.get_trajectory(source_id)
        states = np.asarray(trajectory["observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        if states.shape != (length, 16) or actions.shape != (length, 12):
            raise ValueError(f"Unexpected state/action shape for episode {source_id}")
        for j, index in enumerate(indices):
            writer.add_frame({
                "observation.state": states[index],
                "action": actions[index],
                CAMERA: images[j],
                "task": "open the left drawer",
            })
        writer.save_episode()
        manifest["episodes"].append({"new_episode_index": new_episode_index,
                                     "source_episode_index": int(source_id),
                                     "source_length": length, "target_length": len(indices),
                                     "source_indices": indices})
        print(f"episode {new_episode_index + 1}/21: source={source_id} "
              f"frames={length}->{len(indices)}", flush=True)
    writer.finalize()
    (STAGING / "resampling_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    verify_and_publish()


if __name__ == "__main__":
    main()
