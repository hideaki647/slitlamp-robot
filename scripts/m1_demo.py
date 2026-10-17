"""M1 完成デモ：アームがジョイスティックを操作して、細隙灯を 右眼 → 左眼 → 中央 → 上 → 下 と動かす。

出力：
    outputs/m1_demo.mp4       … 左：横から見たアーム、右：細隙灯カメラ
    outputs/m1_demo_log.csv   … 時刻ごとの 目標位置・実際の位置
    outputs/m1_demo_plot.png  … 目標と実際の位置のグラフ（レポート用）

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m1_demo.py              # 横から
    docker compose run --rm sim python scripts/m1_demo.py --view top   # 真上から → m1_demo_top.mp4
"""
import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim.slitlamp_api import SlitLampRobot
from sim.viz import side_camera, smooth, top_camera, two_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FPS = 30

# (秒数, 目標 x[mm] 始→終, 目標 z[mm] 始→終, 説明)  ※患者の右眼は x=-32mm、左眼は x=+32mm
TIMELINE = [
    (1.0, (0, 0), (0, 0), "grasped joystick"),
    (2.5, (0, -32), (0, 0), "move to patient's RIGHT eye"),
    (1.0, (-32, -32), (0, 0), "hold"),
    (3.5, (-32, 32), (0, 0), "move to patient's LEFT eye"),
    (1.0, (32, 32), (0, 0), "hold"),
    (2.0, (32, 0), (0, 0), "back to center"),
    (2.0, (0, 0), (0, 8), "turn handle: UP 8 mm"),
    (1.0, (0, 0), (8, 8), "hold"),
    (3.0, (0, 0), (8, -8), "turn handle: DOWN 8 mm"),
    (1.0, (0, 0), (-8, -8), "hold"),
    (2.0, (0, 0), (-8, 0), "back to center"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["side", "top"], default="side", help="左画面のカメラ")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    robot = SlitLampRobot(model, data)
    robot.grasp()
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = top_camera() if args.view == "top" else side_camera()
    steps_per_frame = round(1 / FPS / model.opt.timestep)

    log = []  # (時刻, 目標x, 実際x, 目標z, 実際z, 手とノブのずれ) 単位 mm
    path = OUT / ("m1_demo.mp4" if args.view == "side" else f"m1_demo_{args.view}.mp4")
    print(f"{'phase':<32}{'target x':>9}{'x':>8}{'target z':>10}{'z':>8}   [mm]")
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:
        for seconds, (x0, x1), (z0, z1), text in TIMELINE:
            n = int(seconds * FPS)
            for i in range(n):
                s = i / max(n - 1, 1)
                tx, tz = smooth(x0, x1, s), smooth(z0, z1, s)
                robot.command(tx / 1000, tz / 1000)  # 指令（m 単位）→ IK → モーター目標
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

                x, z = (v * 1000 for v in robot.lamp_position())
                log.append((data.time, tx, x, tz, z, robot.weld_error() * 1000))
                writer.append_data(two_panel(
                    renderer, data, cam,
                    [text, f"target x {tx:+5.1f} mm  z {tz:+5.1f} mm",
                     f"lamp   x {x:+5.1f} mm  z {z:+5.1f} mm"],
                    ["slit_cam"]))
            print(f"{text:<32}{tx:>9.1f}{x:>8.2f}{tz:>10.1f}{z:>8.2f}")

    log = np.array(log)
    with open(OUT / "m1_demo_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "target_x_mm", "x_mm", "target_z_mm", "z_mm", "grip_err_mm"])
        w.writerows(np.round(log, 4))

    err = np.hypot(log[:, 1] - log[:, 2], log[:, 3] - log[:, 4])
    print(f"追従誤差（目標と実際の距離）: 平均 {err.mean():.2f} mm, 最大 {err.max():.2f} mm")
    print(f"手とノブのずれ: 最大 {log[:, 5].max():.3f} mm")

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for a, (ti, ai, name) in zip(ax, [(1, 2, "x (left-right)"), (3, 4, "z (height)")]):
        a.plot(log[:, 0], log[:, ti], "--", color="gray", label="target")
        a.plot(log[:, 0], log[:, ai], color="tab:orange", label="actual")
        a.set_ylabel(f"{name} [mm]")
        a.legend(loc="upper right")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "m1_demo_plot.png", dpi=120)
    print("saved", path.relative_to(ROOT), ", m1_demo_log.csv, m1_demo_plot.png")


if __name__ == "__main__":
    main()
