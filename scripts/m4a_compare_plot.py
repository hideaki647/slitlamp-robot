"""M4-A の比較グラフ：先生・モジュール型（校正なし／あり）・生徒3種を、6つの条件で並べる。

前提：outputs/eval_m4a_<方式>_<条件>.json（scripts/eval_policy.py の出力）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4a_compare_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONDS = ["normal", "dark", "bright", "gray_iris", "head_far", "eyeL"]
METHODS = [("expert", "Expert (privileged)", "#7f7f7f"),
           ("modular_raw", "Modular (no calib)", "#b9d7a8"),
           ("modular", "Modular (calibrated)", "#4caf50"),
           ("bc", "E2E: BC (R eye)", "#4c9be8"),
           ("dagger", "E2E: BC+DAgger (R eye)", "#e8794c"),
           ("both", "E2E: BC (both eyes)", "#9b59b6")]


def load(tag, cond):
    f = OUT / f"eval_m4a_{tag}_{cond}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(CONDS))
    w = 0.8 / len(METHODS)
    rows = []
    for i, (tag, name, color) in enumerate(METHODS):
        s = [load(tag, c) for c in CONDS]
        rate = [100 * v["success_rate"] if v else np.nan for v in s]
        err = [v["final_error_mm_mean"] if v else np.nan for v in s]
        pos = x + (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(pos, rate, w, label=name, color=color)
        axes[1].bar(pos, err, w, label=name, color=color)
        rows.append((name, rate, err))

    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("success rate [%]")
    axes[0].set_title("M4-A: End-to-End students vs modular pipeline (50 unseen patients per bar)")
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("final error [mm] (log scale)")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].text(len(CONDS) - 0.5, 1.05, "success threshold 1 mm", ha="right", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["normal", "dark", "bright", "gray iris", "head far", "LEFT eye"])
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / "m4a_compare.png"
    fig.savefig(path, dpi=120)
    print("saved", path.relative_to(ROOT))

    # 表（解説ノートに貼る用）
    print("| method | " + " | ".join(CONDS) + " |")
    for name, rate, err in rows:
        print(f"| {name} | " + " | ".join(f"{r:.0f}% / {e:.3g}" for r, e in zip(rate, err)) + " |")


if __name__ == "__main__":
    main()
