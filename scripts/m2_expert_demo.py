"""M2 デモ動画：先生（特権情報＋P制御）が細隙灯を瞳孔へ自動で合わせる。

左：斜め上からの俯瞰（アーム・細隙灯・患者）、右：細隙灯カメラ（中央の十字＝光軸、緑の円＝合格範囲1mm）。
評価で開始時のずれが大きかった患者条件を選んで、何エピソードか続けて撮る。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m2_expert_demo.py
"""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from expert.expert_policy import expert_action
from sim.env import CONTROL_DT, SUBSTEPS, SUCCESS_MM, SlitLampEnv
from sim.viz import free_camera, label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
W, H = 640, 480
FPS = round(SUBSTEPS / CONTROL_DT)  # 1分割ごとに1コマ → 50 fps（実時間）
N_EPISODES = 4


def slit_view(renderer, data, err_mm, cam_fovy_deg, cam_dist_m,
              title="slit lamp camera (expert sees true pupil position)"):
    """細隙灯カメラの画像に、十字線（光軸）と合格範囲の円、ずれの数値を重ねる。"""
    renderer.update_scene(data, camera="slit_cam")
    img = renderer.render().copy()
    c = (W // 2, H // 2)
    # 1mm が何画素か：画面の縦 H 画素 ＝ 2·距離·tan(fovy/2)
    px_per_mm = H / (2 * cam_dist_m * 1000 * np.tan(np.deg2rad(cam_fovy_deg) / 2))
    ok = np.linalg.norm(err_mm) < SUCCESS_MM
    color = (0, 220, 0) if ok else (255, 255, 255)
    cv2.drawMarker(img, c, color, cv2.MARKER_CROSS, 80, 2)
    cv2.circle(img, c, int(SUCCESS_MM * px_per_mm), (0, 220, 0), 1)
    return label(img, [title,
                       f"error  x {err_mm[0]:+5.1f} mm  z {err_mm[1]:+5.1f} mm  |e| {np.linalg.norm(err_mm):4.1f} mm"
                       + ("  ALIGNED" if ok else "")])


def main():
    OUT.mkdir(exist_ok=True)
    evals = json.loads((OUT / "eval_expert.json").read_text())["episodes"]
    seeds = [e["seed"] for e in sorted(evals, key=lambda e: -e["initial_error_mm"])[:N_EPISODES]]

    env = SlitLampEnv()
    m = env.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    view = free_camera([0.0, -0.05, 0.95], 1.2, 140, -25)
    fovy = m.cam_fovy[m.camera("slit_cam").id]

    path = OUT / "m2_expert_demo.mp4"
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:

        def frame(text):
            d = env.data
            dist = d.site_xpos[env.ids["pupil_site"]][1] - d.cam_xpos[env.ids["cam"]][1]  # カメラ→瞳孔の距離
            renderer.update_scene(d, camera=view)
            left = label(renderer.render(), text)
            writer.append_data(np.hstack([left, slit_view(renderer, d, env.error_mm(), fovy, dist)]))

        for k, seed in enumerate(seeds):
            obs = env.reset(seed)
            head = f"episode {k + 1}/{len(seeds)}  EXPERT (privileged P-control)"
            for _ in range(int(0.7 * FPS)):  # 開始時の状態を少し見せる
                frame([head, "start"])
            hold = 0
            for t in range(40):
                hold = hold + 1 if np.linalg.norm(env.error_mm()) < SUCCESS_MM else 0
                if hold >= 5:
                    break
                a = expert_action(obs, env)
                txt = [head, f"t = {t * CONTROL_DT:.1f} s", f"command dx {a[0]:+4.1f}  dz {a[1]:+4.1f} mm"]
                obs = env.step(a, on_substep=lambda txt=txt: frame(txt))
            for _ in range(int(0.7 * FPS)):  # 合った状態を少し見せる
                frame([head, "aligned"])
    env.close()
    print("saved", path.relative_to(ROOT), "seeds", seeds)


if __name__ == "__main__":
    main()
