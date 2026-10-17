"""M0「試してみよう」：倍率を上げる（ズーム）のと、本体を前に出すのとの違いを動画で比べる。

前半：fovy 15° → 8° → 15°（ズーム。本体は動かない）
後半：lamp_y 0 → +40mm → 0（本体が患者側へ前進。倍率は15°のまま）
左：真横からの俯瞰、右：細隙灯カメラ。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/try_zoom_vs_forward.py
"""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
W, H, FPS = 640, 480, 30


def smooth(a, b, s):
    """a から b へ、なめらかに（始めと終わりをゆっくり）変化させる。s は 0〜1。"""
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


# 各区間：(秒数, fovy の始点→終点, lamp_y の始点→終点[m], 表示する説明)
TIMELINE = [
    (1.0, (15, 15), (0.0, 0.0), "start: fovy 15 deg, lamp_y 0 mm"),
    (3.0, (15, 8), (0.0, 0.0), "ZOOM IN (fovy 15 -> 8): lamp does NOT move"),
    (1.0, (8, 8), (0.0, 0.0), "ZOOM IN (fovy 15 -> 8): lamp does NOT move"),
    (2.0, (8, 15), (0.0, 0.0), "zoom out"),
    (3.0, (15, 15), (0.0, 0.04), "MOVE FORWARD (lamp_y 0 -> +40 mm): zoom unchanged"),
    (1.0, (15, 15), (0.04, 0.04), "MOVE FORWARD (lamp_y 0 -> +40 mm): zoom unchanged"),
    (2.0, (15, 15), (0.04, 0.0), "move back"),
]


def label(img, lines):
    """画像の左上に英字の説明を書く（OpenCV の標準フォントは日本語非対応）。"""
    img = img.copy()
    for i, text in enumerate(lines):
        y = 28 + 26 * i
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = model.camera("slit_cam")
    lamp_x = model.joint("lamp_x").qposadr[0]
    lamp_y = model.joint("lamp_y").qposadr[0]

    # 真横（-x 側）から見るカメラ：前後(y)の動きが左右の動きとして見える
    side = mujoco.MjvCamera()
    side.type = mujoco.mjtCamera.mjCAMERA_FREE
    side.lookat[:] = [0.0, 0.06, 0.98]
    side.distance = 0.75
    side.azimuth = 0
    side.elevation = -10

    data.qpos[lamp_x] = -0.032  # 患者の右眼に合わせておく
    path = OUT / "m0_try_zoom_vs_forward.mp4"
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:
        for seconds, (f0, f1), (y0, y1), text in TIMELINE:
            n = int(seconds * FPS)
            for i in range(n):
                s = i / max(n - 1, 1)
                fovy = smooth(f0, f1, s)
                cam.fovy[0] = fovy
                data.qpos[lamp_y] = smooth(y0, y1, s)
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=side)
                left = label(renderer.render(), [text, f"lamp_y = {data.qpos[lamp_y] * 1000:+.0f} mm"])
                renderer.update_scene(data, camera="slit_cam")
                right = label(renderer.render(), [f"slit_cam  fovy = {fovy:.1f} deg"])
                writer.append_data(np.hstack([left, right]))

    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
