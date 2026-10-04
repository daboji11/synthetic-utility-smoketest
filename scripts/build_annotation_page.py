"""Build a local, unlabeled annotation page from exported preview episodes."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["source", "episode_id", "length", "duration_sec", "instruction", "video_path",
          "valid", "pose_group", "path_group", "placement_group", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="CoffeeSetupMug")
    parser.add_argument("--real-count", type=int, default=10)
    parser.add_argument("--synthetic-count", type=int, default=20)
    args = parser.parse_args()
    output = ROOT / "results/dataset_inspection" / args.task_id
    episodes = list(csv.DictReader((output / "episodes.csv").open(newline="")))
    report = json.loads((output / "inspection.json").read_text())
    episodes.sort(key=lambda row: (row["source"] != "synthetic", row["source"], int(row["episode_id"])))
    if (len(episodes) != args.synthetic_count + args.real_count
            or sum(row["source"] == "synthetic" for row in episodes) != args.synthetic_count
            or sum(row["source"] == "real" for row in episodes) != args.real_count):
        raise ValueError("Preview counts do not match the requested annotation page")

    cards = []
    annotations = []
    for row in episodes:
        source, episode_id = row["source"], row["episode_id"]
        video = Path(row["video_path"]).resolve()
        if video.parent != output / source or not video.is_file():
            raise ValueError(f"Missing or unexpected preview path: {video}")
        fps = report["schemas"][source]["fps"]
        duration = int(row["length"]) / fps
        relative_video = video.relative_to(output).as_posix()
        annotation = {"source": source, "episode_id": episode_id, "length": row["length"],
                      "duration_sec": f"{duration:.3f}", "instruction": row["instruction"],
                      "video_path": relative_video, "valid": "", "pose_group": "",
                      "path_group": "", "placement_group": "", "notes": ""}
        annotations.append(annotation)
        key = f"{source}-{episode_id}"
        cards.append(f"""
        <article class="card" data-source="{html.escape(source)}" data-episode-id="{html.escape(episode_id)}">
          <div class="card-head"><span class="source {html.escape(source)}">{html.escape(source)}</span>
            <strong>Episode {html.escape(episode_id)}</strong><span class="duration">{duration:.1f} s · {html.escape(row['length'])} frames</span></div>
          <video controls preload="metadata" playsinline src="{html.escape(relative_video, quote=True)}" aria-label="{html.escape(source)} episode {html.escape(episode_id)}"></video>
          <p class="instruction">{html.escape(row['instruction'])}</p>
          <div class="fields">
            <label>valid<select data-field="valid" id="valid-{key}"><option value=""></option><option value="yes">yes</option><option value="no">no</option></select></label>
            <label>pose_group<input data-field="pose_group" type="text" autocomplete="off"></label>
            <label>path_group<input data-field="path_group" type="text" autocomplete="off"></label>
            <label>placement_group<input data-field="placement_group" type="text" autocomplete="off"></label>
            <label class="notes">notes<textarea data-field="notes" rows="2"></textarea></label>
          </div>
        </article>""")

    with (output / "annotations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(annotations)

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoffeeSetupMug · Trajectory annotation</title>
<style>
:root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #203029; background: #f3f5f0; }
* { box-sizing: border-box; } body { margin: 0; } header { position: sticky; top: 0; z-index: 1; background: #f3f5f0ee; border-bottom: 1px solid #dce2d8; backdrop-filter: blur(8px); }
.wrap { max-width: 1200px; margin: auto; padding: 20px 24px; } .top { display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
h1 { font-size: 1.65rem; margin: 0 0 5px; } .sub { margin: 0; color: #56665b; font-size: .92rem; } button { border: 0; border-radius: 9px; padding: 11px 16px; background: #245b41; color: white; font: inherit; cursor: pointer; font-weight: 600; } button:hover { background: #18452f; }
main { padding-bottom: 36px !important; } h2 { font-size: 1.18rem; margin: 22px 0 13px; } .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 16px; }
.card { background: white; border: 1px solid #dce2d8; border-radius: 13px; overflow: hidden; box-shadow: 0 2px 7px #233b2910; } .card-head { display: flex; align-items: center; gap: 9px; padding: 13px 15px; } .card-head strong { font-size: .95rem; } .duration { margin-left: auto; color: #637168; font-size: .81rem; white-space: nowrap; }
.source { text-transform: uppercase; letter-spacing: .05em; font-size: .68rem; font-weight: 700; padding: 4px 7px; border-radius: 5px; } .source.synthetic { background: #e6ecff; color: #364f9b; } .source.real { background: #e7f3e9; color: #287147; }
video { display: block; width: 100%; aspect-ratio: 1 / 1; max-height: 360px; background: #151b17; object-fit: contain; } .instruction { padding: 0 15px; margin: 14px 0; font-size: .84rem; color: #516057; min-height: 2.4em; }
.fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 11px; padding: 0 15px 16px; } label { display: flex; flex-direction: column; gap: 5px; font-size: .78rem; font-weight: 650; color: #45554a; } label.notes { grid-column: 1 / -1; } input, select, textarea { width: 100%; padding: 8px 9px; border: 1px solid #cbd5cc; border-radius: 6px; background: white; color: #203029; font: inherit; font-weight: 400; } textarea { resize: vertical; }
@media (max-width: 600px) { .wrap { padding: 16px; } header { position: static; } }
</style></head><body>
<header><div class="wrap top"><div><h1>CoffeeSetupMug</h1><p class="sub">Trajectory annotation · 20 synthetic, then 10 real · Left agent view</p></div><button id="download" type="button">Download annotations.csv</button></div></header>
<main class="wrap"><h2>Synthetic · 20 episodes</h2><section class="grid" aria-label="Synthetic episodes">
""" + "\n".join(cards[:args.synthetic_count]) + "\n</section><h2>Real · 10 episodes</h2><section class=\"grid\" aria-label=\"Real episodes\">\n" + "\n".join(cards[args.synthetic_count:]) + """
</section></main><script>
const metadata = """ + json.dumps(annotations, ensure_ascii=False).replace("</", "<\\/") + """;
const columns = """ + json.dumps(FIELDS) + """;
function csvCell(value) { const s = String(value ?? ""); return /[",\\r\\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s; }
document.getElementById("download").addEventListener("click", () => {
  const rows = metadata.map((entry) => {
    const card = [...document.querySelectorAll(".card")].find((el) => el.dataset.source === entry.source && el.dataset.episodeId === entry.episode_id);
    const updated = {...entry};
    card.querySelectorAll("[data-field]").forEach((el) => { updated[el.dataset.field] = el.value; });
    return columns.map((key) => csvCell(updated[key])).join(",");
  });
  const blob = new Blob(["\\ufeff" + [columns.join(","), ...rows].join("\\r\\n") + "\\r\\n"], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = "annotations.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
</script></body></html>
"""
    page = (page.replace("CoffeeSetupMug", html.escape(args.task_id))
                .replace("20 synthetic", f"{args.synthetic_count} synthetic")
                .replace("10 real", f"{args.real_count} real")
                .replace("Synthetic · 20 episodes", f"Synthetic · {args.synthetic_count} episodes")
                .replace("Real · 10 episodes", f"Real · {args.real_count} episodes"))
    (output / "annotations.html").write_text(page)
    print(f"Created {output / 'annotations.html'} and {output / 'annotations.csv'}")


if __name__ == "__main__":
    main()
