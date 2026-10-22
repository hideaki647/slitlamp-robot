"""M4-H のグラフ：ピント合わせ。手前から始める（臨床でよく行う手順）／奥から始める（近づきすぎ）で、方式を比べる。

前提：outputs/eval_m4h_<方式>_<back|front>.json（scripts/eval_policy.py --focus の出力）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4h_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
METHODS = [("expert", "Expert (privileged)", "#7f7f7f"),
           ("focus1", "Student, 1 frame", "#4c9be8"),
           ("focus3", "Student, 3-frame stack", "#7b3fbf")]
STARTS = [("back", "start: pulled back\n(usual, trained)"), ("front", "start: TOO CLOSE\n(not in training)")]
PANELS = [("success rate [%]", "success_rate", 100),
          ("time to align [s]", "converge_time_s_mean", 1),
          ("first moves away from focus [%]", "focus_away_first3_rate", 100),
          ("final front-back error [mm]", "final_focus_mm_mean", 1)]


def load(tag, start):
    f = OUT / f"eval_m4h_{tag}_{start}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 4.6))
    x = np.arange(len(STARTS))
    w = 0.8 / len(METHODS)
    for ax, (title, key, scale) in zip(axes, PANELS):
        for i, (tag, name, color) in enumerate(METHODS):
            vals = [(s[key] * scale if s and s[key] is not None else np.nan) for s in (load(tag, st) for st, _ in STARTS)]
            pos = x + (i - (len(METHODS) - 1) / 2) * w
            ax.bar(pos, vals, w, color=color, label=name)
            for px, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(px, v, f"{v:.0f}" if scale == 100 else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in STARTS], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("M4-H: focusing (front-back) from images with defocus blur (50 unseen patients each)")
    fig.tight_layout()
    path = OUT / "m4h_compare.png"
    fig.savefig(path, dpi=120)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
