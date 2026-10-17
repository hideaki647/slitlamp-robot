"""M0「試してみよう」：元のファイルは変えずに、読み込んだモデルの値だけを書き換えて比べる。

1. 細隙灯カメラの視野角 fovy を 15° → 8° にする（倍率を上げる）
2. 周回カメラの仰角 elevation を -20° → -60° にする（真上寄りから見下ろす）

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/try_m0.py
"""
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
W, H = 640, 480


def try_fovy(model, data, renderer):
    """fovy=15 と fovy=8 の細隙灯カメラ画像を左右に並べて保存する。"""
    data.qpos[model.joint("lamp_x").qposadr[0]] = -0.032  # 右眼に合わせる
    mujoco.mj_forward(model, data)

    cam = model.camera("slit_cam")
    images = []
    for fovy in (15, 8):
        cam.fovy[0] = fovy  # 読み込み後のモデルの値を直接書き換える
        renderer.update_scene(data, camera="slit_cam")
        images.append(renderer.render())
    cam.fovy[0] = 15  # 元に戻す

    path = OUT / "m0_try_fovy15_vs_8.png"
    imageio.imwrite(path, np.hstack(images))
    print("saved", path.relative_to(ROOT))


def try_elevation(model, data, renderer, seconds=8, fps=30):
    """仰角 -60° で1周する動画を保存する（render_orbit.py と同じ流れ）。"""
    lamp_x = model.joint("lamp_x").qposadr[0]
    orbit = mujoco.MjvCamera()
    orbit.type = mujoco.mjtCamera.mjCAMERA_FREE
    orbit.lookat[:] = [0.0, 0.05, 0.95]
    orbit.distance = 1.3
    orbit.elevation = -60  # ← ここだけ変更（元は -20）

    n = seconds * fps
    path = OUT / "m0_try_elev60_orbit.mp4"
    with imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1) as writer:
        for i in range(n):
            t = i / n
            orbit.azimuth = 360 * t
            data.qpos[lamp_x] = -0.032 * np.cos(2 * np.pi * t)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=orbit)
            left = renderer.render()
            renderer.update_scene(data, camera="slit_cam")
            writer.append_data(np.hstack([left, renderer.render()]))
    print("saved", path.relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    try_fovy(model, data, renderer)
    try_elevation(model, data, renderer)


if __name__ == "__main__":
    main()
