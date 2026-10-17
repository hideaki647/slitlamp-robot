"""M0: カメラがシーンの周りを1周する動画を作る。

左：周回カメラ（俯瞰）、右：細隙灯カメラ。周回の間に細隙灯を右眼→左眼→右眼へ動かす。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/render_orbit.py
"""
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SECONDS = 8
FPS = 30
W, H = 640, 480


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    lamp_x = model.joint("lamp_x").qposadr[0]

    # 自由に動かせるカメラ：注視点(lookat)・距離・方位角(azimuth)・仰角(elevation)で位置を決める
    orbit = mujoco.MjvCamera()
    orbit.type = mujoco.mjtCamera.mjCAMERA_FREE
    orbit.lookat[:] = [0.0, 0.05, 0.95]
    orbit.distance = 1.3
    orbit.elevation = -20  # 少し上から見下ろす

    n_frames = SECONDS * FPS
    path = OUT / "m0_orbit.mp4"
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:
        for i in range(n_frames):
            t = i / n_frames  # 0 → 1
            orbit.azimuth = 360 * t  # 1周

            # 細隙灯を 右眼(-0.032) → 左眼(+0.032) → 右眼 へ、なめらかに往復させる
            data.qpos[lamp_x] = -0.032 * np.cos(2 * np.pi * t)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=orbit)
            left = renderer.render()
            renderer.update_scene(data, camera="slit_cam")
            right = renderer.render()
            writer.append_data(np.hstack([left, right]))

    print("saved", path.relative_to(ROOT), f"({n_frames} frames)")


if __name__ == "__main__":
    main()
