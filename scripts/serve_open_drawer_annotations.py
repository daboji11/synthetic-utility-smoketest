"""Serve cached DreamGen previews and persist manual motion labels to a separate CSV.

Usage: python scripts/serve_open_drawer_annotations.py --port 8765
No GPU, third-party packages, or network downloads are needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "records/opendrawer_dreamgen_video_labels.csv"
OUTPUT = ROOT / "records/opendrawer_dreamgen_motion_annotations.csv"
PAGE = ROOT / "scripts/open_drawer_motion_annotation.html"
PREVIEWS = ROOT / "results/dataset_inspection/OpenDrawer/synthetic"
THREE_CAMERA_PREVIEWS = ROOT / "results/dataset_inspection/OpenDrawer/synthetic_3cam"
CAMERA_KEYS = [
    "observation.images.robot0_agentview_left_image",
    "observation.images.robot0_agentview_right_image",
    "observation.images.robot0_eye_in_hand_image",
]

# Stable CSV values, translated only in the UI.
SCHEMA = {
    "manual_visual_label": ("复核视频结果（可选）", [
        ("success", "成功：视频中抽屉打开"), ("partial", "部分打开"),
        ("failure", "失败"), ("unclear", "无法判断")]),
    "first_motion_direction": ("起手方向", [
        ("left", "向左"), ("right", "向右"), ("up", "向上"), ("down", "向下"),
        ("forward", "向前"), ("backward", "向后"), ("diagonal_left", "斜向左"),
        ("diagonal_right", "斜向右"), ("minimal_direct", "短距离直接接近"),
        ("unclear", "无法判断")]),
    "approach_side": ("从哪里下手", [
        ("from_left", "从左侧"), ("from_right", "从右侧"), ("from_above", "从上方"),
        ("from_below", "从下方"), ("straight_front", "从正面"),
        ("diagonal_left", "左斜侧"), ("diagonal_right", "右斜侧"), ("unclear", "无法判断")]),
    "approach_angle": ("下手角度（画面投影，粗分）", [
        ("near_perpendicular", "接近正对抽屉面"), ("oblique", "斜向接近"),
        ("near_parallel", "接近沿抽屉面横向"), ("unclear", "无法判断")]),
    "handle_contact_position": ("接触把手位置", [
        ("left", "把手左端"), ("center", "把手中部"), ("right", "把手右端"),
        ("unclear", "无法判断")]),
    "contact_quality": ("接触是否可信", [
        ("clear_contact", "明显接触 / 抓住"), ("brief_or_sliding", "短暂接触 / 滑动"),
        ("no_visible_contact", "没有可见接触"), ("geometry_artifact", "穿模 / 几何异常"),
        ("unclear", "无法判断")]),
    "approach_shape": ("接近轨迹", [
        ("direct", "直达"), ("arc_left", "向左绕弧"), ("arc_right", "向右绕弧"),
        ("L_shape", "L 形"), ("multi_stage", "分多段接近"),
        ("zigzag_or_correction", "折返 / 反复修正"), ("unclear", "无法判断")]),
    "pull_direction": ("拉动方向", [
        ("straight", "直拉"), ("left_bias", "偏左"), ("right_bias", "偏右"),
        ("upward_bias", "偏上"), ("downward_bias", "偏下"), ("curved", "弧线"),
        ("unclear", "无法判断")]),
    "pull_style": ("拉动方式", [
        ("single_smooth_pull", "一次平滑拉开"), ("two_stage_pull", "两段拉动"),
        ("multiple_corrections", "多次修正"), ("slow_careful", "缓慢试探"),
        ("fast_direct", "快速直接"), ("unclear", "无法判断")]),
    "final_open_amount": ("最终打开幅度", [
        ("small", "小"), ("medium", "中"), ("large", "大"), ("fully_open", "完全打开"),
        ("unclear", "无法判断")]),
    "trajectory_group": ("自定义轨迹组", [
        ("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("E", "E"),
        ("unassigned", "暂不分组")]),
}
EDITABLE = [*SCHEMA, "motion_reviewed", "notes"]
EXTRA = [*EDITABLE, "motion_review_date", "manual_outcome_reviewed"]
LOCK = threading.RLock()


class Annotations:
    def __init__(self, output: Path):
        self.output = output
        with SOURCE.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self.source_fields = reader.fieldnames
            self.rows = {r["episode_index"]: r for r in reader}
        if len(self.rows) != 60:
            raise ValueError("Expected 60 source episodes")
        for row in self.rows.values():
            video = (ROOT / row["video_path"]).resolve()
            if video.parent != PREVIEWS.resolve() or not video.is_file():
                raise ValueError(f"Missing cached preview: {video}")
            for key in EXTRA:
                row[key] = ""
        manifest_path = THREE_CAMERA_PREVIEWS / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Run scripts/build_open_drawer_three_camera_previews.py first")
        manifest = json.loads(manifest_path.read_text())
        self.previews = {r["episode_index"]: r for r in manifest["previews"]}
        if set(self.previews) != set(self.rows) or manifest["layout"] != ["left", "right", "wrist"]:
            raise ValueError("Unexpected three-camera preview manifest")
        for eid, preview in self.previews.items():
            path = (ROOT / preview["video_path"]).resolve()
            if path.parent != THREE_CAMERA_PREVIEWS.resolve() or not path.is_file():
                raise ValueError(f"Missing three-camera preview: {path}")
            row = self.rows[eid]
            if preview["source_revision"] != row["source_revision"] or preview["frames"] != int(row["source_frames"]):
                raise ValueError(f"Three-camera preview provenance mismatch: {eid}")
        if output.exists():
            with output.open(encoding="utf-8-sig", newline="") as f:
                saved = list(csv.DictReader(f))
            ids = [r["episode_index"] for r in saved]
            if len(ids) != len(set(ids)) or set(ids) != set(self.rows):
                raise ValueError("Saved annotation CSV has unexpected episode IDs")
            for item in saved:
                patch = {k: item.get(k, "") for k in EDITABLE}
                self.validate_patch(patch)
                self.rows[item["episode_index"]].update(
                    {k: item.get(k, "") for k in EXTRA})
        else:
            self.persist()

    def csv_bytes(self):
        f = io.StringIO(newline="")
        writer = csv.DictWriter(f, fieldnames=[*self.source_fields, *EXTRA])
        writer.writeheader()
        writer.writerows(self.rows.values())
        return ("\ufeff" + f.getvalue()).encode("utf-8")

    def revision(self):
        return hashlib.sha256(self.csv_bytes()).hexdigest()

    def persist(self):
        data = self.csv_bytes()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".motion-", dir=self.output.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.output)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def validate_patch(self, patch):
        if not isinstance(patch, dict) or set(patch) - set(EDITABLE):
            raise ValueError("Unknown annotation field")
        for key, value in patch.items():
            if not isinstance(value, str):
                raise ValueError(f"{key}: expected text")
            if key in SCHEMA and value not in ["", *[v for v, _ in SCHEMA[key][1]]]:
                raise ValueError(f"{key}: unknown category {value!r}")
            if key == "motion_reviewed" and value not in ("", "true", "false"):
                raise ValueError("motion_reviewed: expected true / false")
            if key == "notes" and len(value) > 8000:
                raise ValueError("Notes too long")

    def save(self, patches):
        if not isinstance(patches, dict) or not patches or set(patches) - set(self.rows):
            raise ValueError("Unknown or missing episode IDs")
        for patch in patches.values():
            self.validate_patch(patch)
        before = {eid: dict(self.rows[eid]) for eid in patches}
        for eid, patch in patches.items():
            row = self.rows[eid]
            if any(row[k] != v for k, v in patch.items()):
                row.update(patch)
                row["motion_review_date"] = datetime.now(timezone.utc).isoformat()
                row["manual_outcome_reviewed"] = "true" if row["manual_visual_label"] else ""
        try:
            self.persist()
        except Exception:
            for eid, row in before.items():
                self.rows[eid] = row
            raise


def make_handler(db):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/data":
                with LOCK:
                    return self.reply(200, {"rows": list(db.rows.values()),
                        "schema": SCHEMA, "editable": EDITABLE,
                        "revision": db.revision(), "csv_path": str(db.output),
                        "preview_layout": ["left", "right", "wrist"],
                        "preview_camera_keys": CAMERA_KEYS})
            if path == "/annotations.csv":
                with LOCK:
                    return self.reply(200, db.csv_bytes(), "text/csv; charset=utf-8")
            match = re.fullmatch(r"/video/(\d+)", path)
            if match and match[1] in db.rows:
                return self.video(ROOT / db.previews[match[1]]["video_path"])
            return self.reply(404, {"error": "Not found"})

        def video(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    return self.reply(416, {"error": "Invalid range"})
                if not match[1]:
                    start = max(0, size - int(match[2]))
                else:
                    start = int(match[1])
                    end = min(end, int(match[2])) if match[2] else end
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left:
                        data = f.read(min(left, 64 * 1024))
                        if not data:
                            break
                        self.wfile.write(data)
                        left -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != "/api/save":
                return self.reply(404, {"error": "Not found"})
            # Only same-origin browser writes; GET/video remain accessible on localhost.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                return self.reply(403, {"error": "Origin mismatch"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(415, {"error": "Expected application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 1_000_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected object")
                with LOCK:
                    if data.get("revision") != db.revision():
                        return self.reply(409, {"error":
                            "另一页面已更新 CSV。请先导出当前标注，再刷新后合并。"})
                    db.save(data.get("patches"))
                    return self.reply(200, {"revision": db.revision()})
            except (ValueError, TypeError, KeyError) as e:
                return self.reply(400, {"error": str(e)})
            except OSError as e:
                return self.reply(500, {"error": f"CSV save failed: {e}"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    db = Annotations(args.output.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(db))
    print(f"Annotation UI: http://localhost:{args.port}/", flush=True)
    print(f"CSV: {db.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
