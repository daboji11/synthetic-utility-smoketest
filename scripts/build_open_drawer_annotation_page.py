"""Build the OpenDrawer motion-property annotation page from cached previews."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dataset_inspection/OpenDrawer"
FIELDS = ["episode_id", "validity", "first_motion_direction", "approach_side",
          "handle_contact_position", "approach_shape", "pull_direction", "pull_style",
          "final_open_amount", "trajectory_group", "notes"]
OPTIONS = {
    "validity": ["success", "partial", "fail", "unclear"],
    "first_motion_direction": ["left", "right", "up", "down", "forward", "backward",
                               "diagonal_left", "diagonal_right", "minimal_direct", "unclear"],
    "approach_side": ["from_left", "from_right", "from_above", "from_below", "straight_front",
                      "diagonal_left", "diagonal_right", "unclear"],
    "handle_contact_position": ["left", "center", "right", "unclear"],
    "approach_shape": ["direct", "arc_left", "arc_right", "L_shape", "multi_stage",
                       "zigzag_or_correction", "unclear"],
    "pull_direction": ["straight", "left_bias", "right_bias", "upward_bias",
                       "downward_bias", "curved", "unclear"],
    "pull_style": ["single_smooth_pull", "two_stage_pull", "multiple_corrections",
                   "slow_careful", "fast_direct", "unclear"],
    "final_open_amount": ["small", "medium", "large", "fully_open", "unclear"],
    "trajectory_group": ["A", "B", "C", "D", "E", "unassigned"],
}


def select(field: str) -> str:
    options = '<option value="">— choose —</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in OPTIONS[field])
    return (f'<label>{field}<select data-field="{field}" aria-label="{field}">'
            f'{options}</select></label>')


def main() -> None:
    episodes = list(csv.DictReader((OUTPUT / "episodes.csv").open(newline="")))
    report = json.loads((OUTPUT / "inspection.json").read_text())
    synthetic = sorted((row for row in episodes if row["source"] == "synthetic"),
                       key=lambda row: int(row["episode_id"]))
    real = sorted((row for row in episodes if row["source"] == "real"),
                  key=lambda row: int(row["episode_id"]))
    if len(synthetic) != 60 or len(real) != 21:
        raise ValueError(f"Expected 60 synthetic and 21 real previews, got {len(synthetic)} and {len(real)}")
    if len({row["episode_id"] for row in synthetic}) != 60:
        raise ValueError("Duplicate synthetic episode ID")

    def video_info(row: dict) -> tuple[str, str]:
        source = row["source"]
        video = Path(row["video_path"]).resolve()
        if video.parent != OUTPUT / source or not video.is_file():
            raise ValueError(f"Missing or unexpected video: {video}")
        duration = int(row["length"]) / report["schemas"][source]["fps"]
        return video.relative_to(OUTPUT).as_posix(), f"{duration:.1f} s"

    cards = []
    for row in synthetic:
        episode_id = html.escape(row["episode_id"])
        video, duration = video_info(row)
        fields = "".join(select(field) for field in FIELDS[1:-1])
        cards.append(f"""
        <article class="card synthetic-card" data-episode-id="{episode_id}">
          <div class="card-head"><span class="badge">SYNTHETIC</span><strong>Episode {episode_id}</strong><span class="duration">{duration}</span></div>
          <video controls preload="none" playsinline src="{html.escape(video, quote=True)}" aria-label="Synthetic episode {episode_id}"></video>
          <div class="fields">{fields}<label class="notes">notes<textarea data-field="notes" rows="3" aria-label="notes"></textarea></label></div>
        </article>""")

    references = []
    for row in real:
        episode_id = html.escape(row["episode_id"])
        video, duration = video_info(row)
        references.append(f"""
        <article class="reference"><div class="card-head"><span class="badge real">REAL</span><strong>Episode {episode_id}</strong><span class="duration">{duration}</span></div>
          <video controls preload="none" playsinline src="{html.escape(video, quote=True)}" aria-label="Real episode {episode_id}"></video></article>""")

    with (OUTPUT / "annotations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({"episode_id": row["episode_id"]} for row in synthetic)

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenDrawer · Motion annotation</title><style>
:root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #203029; background: #f3f5f0; }
* { box-sizing: border-box; } body { margin: 0; } header { position: sticky; top: 0; z-index: 2; background: #f3f5f0f2; border-bottom: 1px solid #dce2d8; backdrop-filter: blur(8px); }
.wrap { max-width: 1320px; margin: auto; padding: 18px 24px; } .top { display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.18rem; margin: 24px 0 12px; } p { margin: 0; } .sub, .hint { color: #56665b; font-size: .88rem; } .hint { margin: 8px 0 0; }
.controls { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; } button { border: 0; border-radius: 8px; padding: 10px 14px; background: #245b41; color: white; font: inherit; cursor: pointer; font-weight: 600; }
button.secondary { background: #e0e8de; color: #284c35; } button:hover { filter: brightness(.93); } #status { font-size: .8rem; color: #56665b; min-width: 130px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); gap: 16px; } .card, .reference { background: white; border: 1px solid #dce2d8; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 7px #233b2910; }
.card-head { display: flex; align-items: center; gap: 9px; padding: 12px 14px; } .card-head strong { font-size: .95rem; } .duration { margin-left: auto; color: #637168; font-size: .82rem; }
.badge { font-size: .67rem; font-weight: 700; letter-spacing: .05em; padding: 4px 7px; border-radius: 5px; background: #e6ecff; color: #364f9b; } .badge.real { background: #e7f3e9; color: #287147; }
video { display: block; width: 100%; aspect-ratio: 1 / 1; max-height: 360px; background: #151b17; object-fit: contain; }
.fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 14px; } label { display: flex; flex-direction: column; gap: 5px; font-size: .76rem; font-weight: 650; color: #45554a; overflow-wrap: anywhere; } label.notes { grid-column: 1 / -1; }
select, textarea { width: 100%; padding: 8px 9px; border: 1px solid #cbd5cc; border-radius: 6px; background: white; color: #203029; font: inherit; font-weight: 400; } textarea { resize: vertical; }
details { margin: 30px 0; border-top: 1px solid #dce2d8; padding-top: 18px; } summary { cursor: pointer; font-weight: 650; } .reference-grid { margin-top: 16px; }
@media (max-width: 600px) { .wrap { padding: 15px; } header { position: static; } }
</style></head><body>
<header><div class="wrap top"><div><h1>OpenDrawer · Motion annotation</h1><p class="sub">60 synthetic episodes · left agent view · observable motion only</p></div>
<div class="controls"><span id="status" role="status">0 / 60 started</span><button id="save" type="button">Save to localStorage</button><button class="secondary" id="export" type="button">Export CSV</button></div></div></header>
<main class="wrap"><p class="hint">Labels start blank. Use “unclear” where the motion cannot be judged from this view. Groups A–E are provisional motion-pattern groups.</p>
<h2>Synthetic · 60 episodes</h2><section class="grid" aria-label="Synthetic episodes">
""" + "\n".join(cards) + """
</section><details><summary>Real reference videos · 21 episodes (no annotation fields)</summary><section class="grid reference-grid" aria-label="Real reference episodes">
""" + "\n".join(references) + """
</section></details></main><script>
const fields = """ + json.dumps(FIELDS) + """;
const storageKey = "open-drawer-motion-annotations-v1";
const cards = [...document.querySelectorAll(".synthetic-card")];
const status = document.getElementById("status");
function collect() {
  return cards.map((card) => {
    const row = {episode_id: card.dataset.episodeId};
    card.querySelectorAll("[data-field]").forEach((control) => { row[control.dataset.field] = control.value; });
    return row;
  });
}
function countStarted() {
  const started = collect().filter((row) => fields.slice(1).some((field) => row[field])).length;
  status.textContent = `${started} / ${cards.length} started`;
}
cards.forEach((card) => card.addEventListener("input", countStarted));
cards.forEach((card) => card.addEventListener("change", countStarted));
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
  if (saved && typeof saved === "object") {
    cards.forEach((card) => {
      const row = saved[card.dataset.episodeId];
      if (!row) return;
      card.querySelectorAll("[data-field]").forEach((control) => {
        const value = row[control.dataset.field];
        if (typeof value === "string") control.value = value;
      });
    });
    countStarted();
    status.title = "Restored from localStorage";
  }
} catch (error) { status.title = `localStorage unavailable: ${error.message}`; }
document.getElementById("save").addEventListener("click", () => {
  try {
    const data = Object.fromEntries(collect().map((row) => [row.episode_id, row]));
    localStorage.setItem(storageKey, JSON.stringify(data));
    countStarted();
    status.title = "Saved to localStorage";
    alert("Saved to localStorage on this browser.");
  } catch (error) { alert(`Could not save locally: ${error.message}. Export CSV to keep a copy.`); }
});
function csvCell(value) { const s = String(value ?? ""); return /[",\\r\\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s; }
document.getElementById("export").addEventListener("click", () => {
  const rows = collect().map((row) => fields.map((field) => csvCell(row[field])).join(","));
  const blob = new Blob(["\\ufeff" + [fields.join(","), ...rows].join("\\r\\n") + "\\r\\n"], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = "open_drawer_annotations.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
</script></body></html>
"""
    (OUTPUT / "annotations.html").write_text(page)
    print(f"Created {OUTPUT / 'annotations.html'} and {OUTPUT / 'annotations.csv'}")


if __name__ == "__main__":
    main()
