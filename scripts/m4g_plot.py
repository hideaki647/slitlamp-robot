"""M4-G のグラフ：遅れ・指令のばらつき・モーターの個体差・画像の質 に対する、4方式の強さ。

前提：outputs/eval_m4g_<方式>_<条件>.json（scripts/eval_policy.py の出力）。
遅れの図には、比例制御の理論（次のずれ = 今のずれ − 0.5 × kステップ前のずれ）から求めた
「ずれが1ステップで何倍になるか」（特性方程式の根の大きさの最大値。1を超えると発散）を重ねる。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4g_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
METHODS = [("expert", "Expert (privileged)", "#7f7f7f", "s"),
           ("modular", "Modular (pupil detection)", "#4caf50", "^"),
           ("bc", "E2E: BC", "#4c9be8", "o"),
           ("dagger", "E2E: BC+DAgger", "#e8794c", "D")]
PANELS = [
    ("observation delay [s]", [("base", 0.0), ("delay1", 0.1), ("delay2", 0.2), ("delay3", 0.3)]),
    ("command noise std [mm]", [("base", 0.0), ("act0.1", 0.1), ("act0.3", 0.3), ("act0.5", 0.5)]),
    ("motor strength kp (x nominal)", [("kp0.5", 0.5), ("kp0.7", 0.7), ("base", 1.0), ("kp1.3", 1.3), ("kp1.5", 1.5)]),
    ("image quality", [("base", "clean"), ("noise0.02", "noise 2%"), ("noise0.05", "noise 5%"),
                       ("blur0.5", "blur 0.5px"), ("blur1", "blur 1px")]),
]
GAIN = 0.5


def theory_root(k):
    """遅れ k ステップの比例制御 e[n+1] = e[n] − GAIN·e[n−k] の、根の大きさの最大値。"""
    c = np.zeros(k + 2)
    c[0], c[1] = 1, -1
    c[-1] += GAIN
    return max(abs(np.roots(c)))


def load(tag, name):
    f = OUT / f"eval_m4g_{tag}_{name}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    table = []
    for col, (title, levels) in enumerate(PANELS):
        xs = np.arange(len(levels))
        for tag, label, color, marker in METHODS:
            s = [load(tag, n) for n, _ in levels]
            rate = [100 * v["success_rate"] if v else np.nan for v in s]
            err = [v["final_error_mm_mean"] if v else np.nan for v in s]
            osc = [v["oscillations_mean"] if v else np.nan for v in s]
            kw = dict(color=color, marker=marker, label=label)
            axes[0, col].plot(xs, rate, **kw)
            axes[1, col].plot(xs, err, **kw)
            axes[2, col].plot(xs, osc, **kw)
            table.append((title, label, [f"{r:.0f}%/{e:.3g}mm/{o:.1f}" for r, e, o in zip(rate, err, osc)]))
        for row in range(3):
            ax = axes[row, col]
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for _, v in levels], fontsize=8)
            ax.grid(alpha=0.3)
        axes[0, col].set_title(title)
        axes[0, col].set_ylim(-3, 105)
        axes[1, col].set_yscale("log")
        axes[1, col].axhline(1.0, color="k", ls=":", lw=1)
        axes[2, col].set_xlabel(title)

    # 遅れの図に理論を重ねる（右の目盛り）
    ax = axes[0, 0].twinx()
    ks = [0, 1, 2, 3]
    ax.plot(ks, [theory_root(k) for k in ks], "k--", lw=1.5, label="theory: max |root| (>1 = unstable)")
    ax.axhline(1.0, color="r", lw=0.8, ls=":")
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("theory: error multiplier per step", fontsize=8)
    ax.legend(loc="lower left", fontsize=7)

    axes[0, 0].set_ylabel("success rate [%]")
    axes[1, 0].set_ylabel("final error [mm] (log)")
    axes[2, 0].set_ylabel("oscillations (direction changes)")
    axes[0, 3].legend(fontsize=8, loc="lower right")
    fig.suptitle("M4-G: robustness to real-world imperfections (50 unseen patients per point, no retraining)")
    fig.tight_layout()
    path = OUT / "m4g_robustness.png"
    fig.savefig(path, dpi=110)
    print("saved", path.relative_to(ROOT))
    for title, label, cells in table:
        print(f"{title} | {label} | " + " | ".join(cells))
    print("theory", {k: round(theory_root(k), 3) for k in range(5)})


if __name__ == "__main__":
    main()
