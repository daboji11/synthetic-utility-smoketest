"""Metadata-first adapter for LeRobot v3 RoboCasa datasets on Hugging Face.

The adapter downloads only info, task and episode metadata at construction.
Trajectory arrays and shared video files are fetched lazily when requested.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import re

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from .base_dataset import BaseRobotDataset, Trajectory


class RoboCasaLeRobotDataset(BaseRobotDataset):
    def __init__(self, repo_id: str, task_id: str | None = None,
                 camera_key: str | None = None, source: str | None = None,
                 task_label: str | None = None, task_pattern: str | None = None,
                 cache_dir: str | Path = "data/raw/huggingface", offline: bool = False):
        self.repo_id = repo_id
        self.source = source or "unknown"
        self.task_aliases = {task_id: task_label} if task_id and task_label else {}
        self.task_pattern = re.compile(task_pattern) if task_id and task_pattern else None
        self.canonical_task_id = task_id
        self.cache_dir = Path(cache_dir).resolve()
        self.offline = offline
        if offline:
            snapshots = [path for path in (self.cache_dir / f"datasets--{repo_id.replace('/', '--')}" / "snapshots").glob("*")
                         if path.is_dir()]
            if len(snapshots) != 1:
                raise ValueError(f"Expected one cached snapshot for {repo_id}, found {len(snapshots)}")
            self.snapshot = snapshots[0]
            self.revision = self.snapshot.name
        else:
            self.revision = HfApi().dataset_info(repo_id).sha
        self.info = self._read_json("meta/info.json")
        self.features = self.info["features"]
        self.camera_keys = sorted(key for key, feature in self.features.items()
                                  if feature.get("dtype") == "video")
        self.camera_key = self._resolve_camera_key(camera_key)
        tasks_table = pq.read_table(self._download("meta/tasks.parquet"))
        self.task_names = {int(row["task_index"]): row.get("__index_level_0__", str(row["task_index"]))
                           for row in tasks_table.to_pylist()}
        if offline:
            episode_files = sorted(str(path.relative_to(self.snapshot))
                                   for path in (self.snapshot / "meta/episodes").glob("**/*.parquet"))
        else:
            episode_files = sorted(path for path in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=self.revision)
                                   if path.startswith("meta/episodes/") and path.endswith(".parquet"))
        if not episode_files:
            raise ValueError(f"No LeRobot v3 episode metadata in {repo_id}")
        self._all_episodes = [row for path in episode_files
                              for row in pq.read_table(self._download(path)).to_pylist()]
        self._episodes = [row for row in self._all_episodes
                          if task_id is None or self._matches_task(row, task_id)]
        self._by_id = {str(row["episode_index"]): row for row in self._episodes}
        if len(self._by_id) != len(self._episodes):
            raise ValueError("Duplicate episode_index in metadata")
        self._data_tables: dict[str, Any] = {}

    def _download(self, path: str) -> Path:
        if self.offline:
            cached = self.snapshot / path
            if not cached.is_file():
                raise FileNotFoundError(f"Required cached file is missing: {cached}")
            return cached
        return Path(hf_hub_download(repo_id=self.repo_id, filename=path, repo_type="dataset",
                                    revision=self.revision, cache_dir=self.cache_dir))

    def _read_json(self, path: str) -> dict:
        import json
        return json.loads(self._download(path).read_text())

    def _resolve_camera_key(self, camera_key: str | None) -> str | None:
        if camera_key is None:
            return None
        matches = [key for key in self.camera_keys if key == camera_key or key.endswith("." + camera_key)]
        if len(matches) != 1:
            raise ValueError(f"Camera {camera_key!r} not found uniquely; available: {self.camera_keys}")
        return matches[0]

    def _label(self, task_id: str) -> str:
        return self.task_aliases.get(task_id, task_id)

    def _canonical(self, label: str) -> str:
        if self.task_pattern and self.task_pattern.fullmatch(label):
            return self.canonical_task_id
        return next((task_id for task_id, value in self.task_aliases.items() if value == label), label)

    def _matches_task(self, episode: dict, task_id: str) -> bool:
        if task_id == self.canonical_task_id and self.task_pattern:
            return any(self.task_pattern.fullmatch(label) for label in episode["tasks"])
        return self._label(task_id) in episode["tasks"]

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Trajectory:
        return self.get_trajectory(str(self._episodes[index]["episode_index"]))

    def get_episode_ids(self) -> list[str]:
        return [str(row["episode_index"]) for row in self._episodes]

    def get_episode_metadata(self, episode_id: str | int) -> dict:
        return dict(self._by_id[str(episode_id)])

    def get_task_ids(self) -> list[str]:
        return sorted({self._canonical(task) for episode in self._episodes for task in episode["tasks"]})

    def episode_counts_by_task(self) -> dict[str, int]:
        return dict(Counter(self._canonical(task) for episode in self._episodes for task in episode["tasks"]))

    def filter_by_task(self, task_id: str) -> RoboCasaLeRobotDataset:
        filtered = object.__new__(type(self))
        filtered.__dict__ = self.__dict__.copy()
        filtered._episodes = [row for row in self._episodes if self._matches_task(row, task_id)]
        filtered._by_id = {str(row["episode_index"]): row for row in filtered._episodes}
        return filtered

    def get_trajectory(self, trajectory_id: str) -> Trajectory:
        episode = self.get_episode_metadata(trajectory_id)
        path = self.info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                              file_index=episode["data/file_index"])
        if path not in self._data_tables:
            self._data_tables[path] = pq.read_table(self._download(path),
                                                     columns=["episode_index", "observation.state", "action"])
        table = self._data_tables[path]
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        rows = table.slice(start, end - start)
        if rows.num_rows != episode["length"]:
            raise ValueError(f"Episode length mismatch for {trajectory_id}")
        if set(rows.column("episode_index").to_pylist()) != {int(trajectory_id)}:
            raise ValueError(f"Episode index mismatch for {trajectory_id}")
        return {"trajectory_id": str(trajectory_id), "observations": rows.column("observation.state").to_pylist(),
                "actions": rows.column("action").to_pylist(), "task_id": self._canonical(episode["tasks"][0]),
                "success": None, "metadata": {"source": self.source, "repo_id": self.repo_id,
                                              "camera_key": self.camera_key, "episode_index": int(trajectory_id),
                                              "tasks": episode["tasks"]}}

    def video_segment(self, episode_id: str | int, camera_key: str | None = None) -> tuple[Path, float, float]:
        episode = self.get_episode_metadata(episode_id)
        key = self._resolve_camera_key(camera_key) if camera_key else self.camera_key
        if key is None:
            raise ValueError("Select a camera key before exporting video")
        prefix = f"videos/{key}"
        path = self.info["video_path"].format(video_key=key,
                                               chunk_index=episode[f"{prefix}/chunk_index"],
                                               file_index=episode[f"{prefix}/file_index"])
        start = float(episode[f"{prefix}/from_timestamp"])
        end = float(episode[f"{prefix}/to_timestamp"])
        if end <= start:
            raise ValueError(f"Invalid video timestamps for episode {episode_id}")
        return self._download(path), start, end - start
