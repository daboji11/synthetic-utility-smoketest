"""Export synchronized left/right/wrist DreamGen previews from cached shards.

Run in the existing data environment (pyarrow required):
python scripts/build_open_drawer_three_camera_previews.py
"""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "records/opendrawer_dreamgen_video_labels.csv"
OUTPUT = ROOT / "results/dataset_inspection/OpenDrawer/synthetic_3cam"
CAMERAS = [
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
]


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration",
        "-of", "json", str(path)], text=True))["streams"][0]


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    revs = {r["source_revision"] for r in rows}
    assert len(revs) == 1
    revision = revs.pop()
    snapshot = ROOT / ("data/raw/huggingface/"
        "datasets--Keh0t0--dreamgen_robocasa_p3_init0_v2_droid/snapshots") / revision
    info = json.loads((snapshot / "meta/info.json").read_text())
    fps = info["fps"]
    episodes = {
        str(r["episode_index"]): r
        for p in (snapshot / "meta/episodes").rglob("*.parquet")
        for r in pq.read_table(p).to_pylist()
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def export(row):
        eid = row["episode_index"]
        meta = episodes[eid]
        assert meta["tasks"] == [row["instruction"]]
        count = meta["length"]
        assert count == int(row["source_frames"])
        inputs, ranges = [], []
        for camera in CAMERAS:
            prefix = "videos/" + camera + "/"
            start, end = meta[prefix + "from_timestamp"], meta[prefix + "to_timestamp"]
            assert abs((end - start) * fps - count) < 1e-6
            video = snapshot / info["video_path"].format(
                video_key=camera, chunk_index=meta[prefix + "chunk_index"],
                file_index=meta[prefix + "file_index"])
            assert video.is_file(), video
            inputs += ["-ss", f"{start:.9f}", "-threads", "1", "-i", str(video)]
            ranges.append({"camera_key": camera, "source_video": str(video.relative_to(ROOT)),
                           "from_timestamp": start, "to_timestamp": end})
        assert abs(ranges[0]["from_timestamp"] - float(row["source_from_timestamp"])) < 1e-6
        destination = OUTPUT / f"synthetic_episode_{eid}_3cam.mp4"
        temporary = destination.with_suffix(".building.mp4")
        try:
            if not destination.exists():
                filters = ";".join(f"[{i}:v]trim=end_frame={count},setpts=PTS-STARTPTS[v{i}]"
                                   for i in range(3))
                filters += ";[v0][v1][v2]hstack=inputs=3:shortest=1[out]"
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex_threads", "1", "-filter_complex", filters,
                    "-map", "[out]", "-an", "-frames:v", str(count),
                    "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True)
                stats = probe(temporary)
                if (stats["width"], stats["height"], int(stats["nb_frames"])) != (384, 128, count):
                    raise ValueError(f"Unexpected preview geometry/length: {stats}")
                os.replace(temporary, destination)
            stats = probe(destination)
            assert (stats["width"], stats["height"], int(stats["nb_frames"])) == (384, 128, count)
            rate_num, rate_den = map(int, stats["avg_frame_rate"].split("/"))
            assert rate_num / rate_den == fps
            entry = {"episode_index": eid, "source_revision": revision,
                "layout": ["left", "right", "wrist"], "fps": fps,
                "frames": count, "video_path": destination.relative_to(ROOT).as_posix(),
                "video_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "camera_segments": ranges}
            print("PREVIEW_READY", eid, f"{count} frames, 384x128", flush=True)
            return entry
        finally:
            if temporary.exists():
                temporary.unlink()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(export, rows))
    manifest = {"source_repo": rows[0]["source_repo"], "source_revision": revision,
                "episodes": len(entries), "layout": ["left", "right", "wrist"],
                "fps": fps, "previews": entries}
    temporary = OUTPUT / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, OUTPUT / "manifest.json")
    print("THREE_CAMERA_PREVIEWS_COMPLETE", len(entries), flush=True)


if __name__ == "__main__":
    main()
