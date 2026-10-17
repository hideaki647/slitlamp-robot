"""M3 の結果のグラフ：①DAgger のラウンドごとの成功率 ②先生・BC・BC+DAgger を条件別に比較。

前提：outputs/dagger_progress.json と、eval_<方式>[_<条件>].json（scripts/eval_policy.py の出力）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m3_compare_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.env import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
POLICIES = [("expert", "Expert (privileged)"), ("bc", "Student: BC"), ("dagger", "Student: BC+DAgger")]
COLORS = ["#7f7f7f", "#4c9be8", "#e8794c"]


def load(tag, cond):
    f = OUT / (f"eval_{tag}.json" if cond == "normal" else f"eval_{tag}_{cond}.json")
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ① DAgger の学習曲線（成功率はどのラウンドも100%なので、最終誤差で見る）
    prog = json.loads((OUT / "dagger_progress.json").read_text())
    ax = axes[0]
    ax.plot([p["round"] for p in prog], [p["final_error_mm_mean"] for p in prog], "o-", color=COLORS[2])
    ax.axhline(load("expert", "normal")["final_error_mm_mean"], color=COLORS[0], ls="--", label="expert")
    ax.set_xticks([p["round"] for p in prog])
    ax.set_xticklabels(["BC"] + [f"R{p['round']}" for p in prog[1:]])
    ax.set_ylim(0, None)
    ax.set_ylabel("final error [mm] (50 unseen patients)")
    ax.set_title("DAgger rounds (success 100% in every round)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ② 条件別の成功率、③ 条件別の収束時間
    x = np.arange(len(CONDITIONS))
    w = 0.27
    for i, (tag, name) in enumerate(POLICIES):
        s = [load(tag, c) for c in CONDITIONS]
        rate = [100 * v["success_rate"] if v else np.nan for v in s]
        conv = [v["converge_time_s_mean"] if v and v["converge_time_s_mean"] is not None else np.nan for v in s]
        axes[1].bar(x + (i - 1) * w, rate, w, label=name, color=COLORS[i])
        axes[2].bar(x + (i - 1) * w, conv, w, label=name, color=COLORS[i])
    for ax, title, yl in [(axes[1], "success rate by condition", "success rate [%]"),
                          (axes[2], "convergence time (successful episodes)", "time to < 1 mm [s]")]:
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS)
        ax.set_title(title)
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    path = OUT / "m3_compare.png"
    fig.savefig(path, dpi=120)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
