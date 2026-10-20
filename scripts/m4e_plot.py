"""M4-E のグラフ：瞬きに対する方式ごとの強さ。

前提：outputs/eval_m4e_<方式>_rest.json（瞼はあるが瞬きしない）と eval_m4e_<方式>_eval.json（1.0秒・2.5秒に瞬き）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4e_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
METHODS = [("expert", "Expert (waits)", "#7f7f7f"),
           ("modular", "Modular", "#4caf50"),
           ("bc", "E2E BC (no blinks)", "#4c9be8"),
           ("dagger", "E2E BC+DAgger (no blinks)", "#e8794c"),
           ("blink1", "E2E trained w/ blinks (1 frame)", "#c0a0e0"),
           ("blink3", "E2E trained w/ blinks (3-frame stack)", "#7b3fbf")]


def load(tag, kind):
    f = OUT / f"eval_m4e_{tag}_{kind}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    panels = [("movement during blink (mean of per-patient max) [mm]", "eval", "blink_move_max_mm_mean"),
              ("worst movement during blink [mm]", "eval", "blink_move_max_mm_max"),
              ("aligned at the end [%]", "eval", "final_aligned_rate"),
              ("final error, lid present but no blink [mm]", "rest", "final_error_mm_mean")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, (title, kind, key) in zip(axes, panels):
        vals = []
        for tag, name, color in METHODS:
            s = load(tag, kind)
            v = s[key] if s else np.nan
            vals.append(100 * v if key == "final_aligned_rate" else v)
        ax.bar(range(len(METHODS)), vals, color=[c for _, _, c in METHODS])
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([n for _, n, _ in METHODS], rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}" if key != "final_aligned_rate" else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("M4-E: blinks at 1.0 s and 2.5 s (50 unseen patients, 5 s episodes)")
    fig.tight_layout()
    path = OUT / "m4e_compare.png"
    fig.savefig(path, dpi=120)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
