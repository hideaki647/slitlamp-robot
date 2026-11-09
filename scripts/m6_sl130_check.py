"""M6-①：SL130 の見た目の確認画像。実機の写真と同じ向き（右側面・患者側・検者側・斜め）と、倍率ドラムの拡大。

アームがジョイスティックを握った状態で撮る（部品とアームが重ならないかの確認も兼ねる）。
--drum で倍率ドラムを回した角度 [度] を指定できる（+60° ごとに 12 → 20 → 32 → 12 → 8 → 5）。
--video で、倍率ドラムを +60° ずつ1周回す拡大動画（outputs/m6_drum_turn.mp4）を作る。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m6_sl130_check.py
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from sim.slitlamp_api import SlitLampRobot

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 480
# (名前, 見る点, 距離, 方位角, 仰角, 患者の頭を隠すか)。方位角 0°＝-x 側から +x の向きに見る、180°＝+x 側から
# 見る点が文字列のものは、その部品（倍率ドラム sl_drum_R / sl_drum_L）の位置を見る
VIEWS = [("right side (photo 2)", (-0.03, -0.06, 0.98), 0.75, 180, -5, False),
         ("patient side (photo 3)", (-0.03, -0.06, 0.98), 0.85, -90, -8, True),
         ("examiner side", (-0.03, -0.06, 0.98), 0.85, 90, -15, False),
         ("oblique", (-0.03, -0.06, 0.98), 0.85, 215, -20, False),
         ("drum close-up (right)", "sl_drum_R", 0.14, 180, 0, False),
         ("drum close-up (left)", "sl_drum_L", 0.14, 0, 0, False)]


def setup(drum_deg=0.0):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene_sl130.xml"))
    data = mujoco.MjData(model)
    robot = SlitLampRobot(model, data)
    robot.grasp(-0.032, 0.0)                        # 患者の右眼の前で握る
    data.qpos[model.joint("mag_drum").qposadr[0]] = np.radians(drum_deg)
    mujoco.mj_forward(model, data)
    return model, data


def render_views(model, data):
    r = mujoco.Renderer(model, height=H, width=W)
    tiles = []
    head = model.body("head").id
    head_pos = model.body_pos[head].copy()
    for name, lookat, dist, az, el, hide_head in VIEWS:
        model.body_pos[head] = head_pos + (np.array([0, 2.0, 0]) if hide_head else 0)   # 写真には患者がいないので遠くへ
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = data.geom_xpos[model.geom(lookat).id] if isinstance(lookat, str) else lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(data, cam)
        img = r.render().copy()
        cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(img)
    model.body_pos[head] = head_pos
    return np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])


def drum_video(model, data, path):
    """右側面のドラムを拡大し、+60° ずつ1周回す（1段 0.5秒で回し、0.7秒止まる）。"""
    r = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.geom_xpos[model.geom("sl_drum_R").id]
    cam.distance, cam.azimuth, cam.elevation = 0.14, 180, 0
    qadr = model.joint("mag_drum").qposadr[0]
    order = ["12", "20", "32", "12", "8", "5", "12"]
    with imageio.get_writer(ROOT / path, fps=30, quality=8, macro_block_size=1) as w:
        for k in range(6):
            for t in np.r_[np.linspace(0, 1, 15), np.ones(21)]:
                data.qpos[qadr] = np.radians(60 * (k + t))
                mujoco.mj_forward(model, data)
                r.update_scene(data, cam)
                img = r.render().copy()
                label = order[k + 1] if t >= 1 else order[k]
                cv2.putText(img, f"x{label}", (W - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
                w.append_data(img)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drum", type=float, default=0.0)
    p.add_argument("--out", default="outputs/m6_sl130_check.png")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    model, data = setup(args.drum)
    if args.video:
        drum_video(model, data, "outputs/m6_drum_turn.mp4")
        return
    imageio.imwrite(ROOT / args.out, render_views(model, data))
    print("saved", args.out)


if __name__ == "__main__":
    main()
