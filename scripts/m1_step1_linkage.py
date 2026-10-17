"""M1 Step1：ジョイスティックを仮モーターで直接動かし、連動拘束で本体がついてくるか確かめる。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m1_step1_linkage.py
"""
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from sim.viz import side_camera, smooth, two_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FPS = 30

# (秒数, 左右傾き[deg] 始→終, 回転[deg] 始→終, 説明)
TIMELINE = [
    (1.0, (0, 0), (0, 0), "start"),
    (2.0, (0, 10), (0, 0), "tilt RIGHT 10 deg"),
    (1.0, (10, 10), (0, 0), "hold"),
    (3.0, (10, -10), (0, 0), "tilt LEFT -10 deg"),
    (1.0, (-10, -10), (0, 0), "hold"),
    (2.0, (-10, 0), (0, 0), "back to center"),
    (2.0, (0, 0), (0, 180), "spin handle +180 deg (up)"),
    (1.0, (0, 0), (180, 180), "hold"),
    (3.0, (0, 0), (180, -180), "spin handle -180 deg (down)"),
    (1.0, (0, 0), (-180, -180), "hold"),
]


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "test_linkage.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = side_camera()

    q = {n: model.joint(n).qposadr[0] for n in ("js_roll", "js_spin", "lamp_x", "lamp_z")}
    steps_per_frame = round(1 / FPS / model.opt.timestep)

    print(f"{'phase':<30}{'js_roll[deg]':>13}{'lamp_x[mm]':>12}{'ratio[m/rad]':>14}"
          f"{'js_spin[deg]':>14}{'lamp_z[mm]':>12}{'ratio[m/rad]':>14}")
    path = OUT / "m1_step1_linkage.mp4"
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:
        for seconds, (r0, r1), (s0, s1), text in TIMELINE:
            n = int(seconds * FPS)
            for i in range(n):
                s = i / max(n - 1, 1)
                # 仮モーターに目標角度（ラジアン）を渡す
                data.ctrl[model.actuator("test_roll").id] = np.deg2rad(smooth(r0, r1, s))
                data.ctrl[model.actuator("test_spin").id] = np.deg2rad(smooth(s0, s1, s))
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)  # 物理を 2ms ずつ進める（ここで拘束が効く）

                roll, spin = data.qpos[q["js_roll"]], data.qpos[q["js_spin"]]
                x, z = data.qpos[q["lamp_x"]], data.qpos[q["lamp_z"]]
                writer.append_data(two_panel(
                    renderer, data, cam,
                    [text, f"joystick roll {np.rad2deg(roll):+5.1f} deg  spin {np.rad2deg(spin):+6.1f} deg",
                     f"lamp x {x * 1000:+5.1f} mm  z {z * 1000:+5.1f} mm"],
                    ["slit_cam"]))

            # 各区間の終わりの値を表にする。比が 0.15 と 0.00318 になっていれば正しく連動している
            rx = x / roll if abs(roll) > 1e-3 else float("nan")
            rz = z / spin if abs(spin) > 1e-3 else float("nan")
            print(f"{text:<30}{np.rad2deg(roll):>13.2f}{x * 1000:>12.2f}{rx:>14.4f}"
                  f"{np.rad2deg(spin):>14.1f}{z * 1000:>12.2f}{rz:>14.5f}")

    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
