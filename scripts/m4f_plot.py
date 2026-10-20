"""M4-F のグラフ：両眼切替（右眼→左眼、左眼→右眼）での、方式ごとの成功率・到達時間・最終誤差。

前提：outputs/eval_m4f_<方式>_to<L|R>.json（scripts/eval_policy.py --task switch の出力）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4f_plot.py
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
           ("baseline", "Fixed 64 mm move + modular", "#4caf50"),
           ("both_nogoal", "E2E student, no instruction", "#9b59b6"),
           ("goal", "E2E student + instruction", "#e8794c")]
DIRS = [("toL", "right -> LEFT eye"), ("toR", "left -> RIGHT eye")]


def load(tag, d):
    f = OUT / f"eval_m4f_{tag}_{d}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(DIRS))
    w = 0.8 / len(METHODS)
    for i, (tag, name, color) in enumerate(METHODS):
        s = [load(tag, d) for d, _ in DIRS]
        pos = x + (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(pos, [100 * v["success_rate"] for v in s], w, color=color, label=name)
        axes[1].bar(pos, [v["first_in_s_mean"] or np.nan for v in s], w, color=color)
        axes[2].bar(pos, [v["final_error_mm_mean"] for v in s], w, color=color)
    for ax, title, yl in [(axes[0], "success rate", "[%]"),
                          (axes[1], "time to reach target eye (< 1 mm)", "[s]"),
                          (axes[2], "final error (log)", "[mm]")]:
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in DIRS])
        ax.set_title(title)
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 105)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("M4-F: switching eyes (50 unseen patients each, IPD 56-72 mm, start aligned to the other eye)")
    fig.tight_layout()
    path = OUT / "m4f_compare.png"
    fig.savefig(path, dpi=120)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
