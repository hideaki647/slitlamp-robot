"""M4-F 動画：指示つきの生徒が、右眼 → 左眼 → 右眼 と移って合わせる。

左：俯瞰、右：細隙灯カメラ（十字＝光軸、合うと緑）。画面に「指示」を表示する。右下の黄色い枠＝生徒が見ている画像。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4f_switch_demo.py --policy student:checkpoints/bc_goal.pt
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from scripts.eval_policy import EVAL_SEED0, load_policy
from scripts.m2_expert_demo import FPS, H, W, slit_view
from sim.env import CONTROL_DT, SUCCESS_MM, SlitLampEnv
from sim.viz import free_camera, label

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="student:checkpoints/bc_goal.pt")
    p.add_argument("--name", default="STUDENT with instruction (image + which eye)")
    p.add_argument("--seed", type=int, default=EVAL_SEED0 + 3)
    p.add_argument("--out", default="outputs/m4f_switch_demo.mp4")
    args = p.parse_args()

    policy = load_policy(args.policy)
    env = SlitLampEnv(target_eye="R", ipd_mm=(56, 72))
    m = env.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    view = free_camera([0.0, -0.05, 0.95], 1.2, 140, -25)
    fovy = m.cam_fovy[m.camera("slit_cam").id]
    box = {"img": None}

    path = ROOT / args.out
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:

        def frame(text):
            d = env.data
            dist = d.site_xpos[env.ids["pupil_site"]][1] - d.cam_xpos[env.ids["cam"]][1]
            renderer.update_scene(d, camera=view)
            left = label(renderer.render(), text)
            right = slit_view(renderer, d, env.error_mm(), fovy, dist,
                              title=f"slit lamp camera   instruction: {'RIGHT' if env.goal == 0 else 'LEFT'} eye")
            inset = cv2.resize(box["img"], (192, 192), interpolation=cv2.INTER_NEAREST)
            right[H - 200:H - 8, W - 200:W - 8] = inset
            cv2.rectangle(right, (W - 201, H - 201), (W - 7, H - 7), (255, 255, 0), 1)
            writer.append_data(np.hstack([left, right]))

        obs = env.reset(args.seed)
        box["img"] = obs
        for eye in ["R", "L", "R"]:
            env.set_goal(eye)
            head = f"{args.name}   IPD {env.info['ipd_mm']:.0f} mm"
            for _ in range(int(0.6 * FPS)):
                frame([head, f"instruction -> {'RIGHT' if eye == 'R' else 'LEFT'} eye"])
            hold = 0
            for t in range(80):
                hold = hold + 1 if np.linalg.norm(env.error_mm()) < SUCCESS_MM else 0
                if hold >= 5:
                    break
                a = policy(obs, env)
                txt = [head, f"instruction -> {'RIGHT' if eye == 'R' else 'LEFT'} eye   t = {t * CONTROL_DT:.1f} s",
                       f"command dx {a[0]:+4.1f}  dz {a[1]:+4.1f} mm"]
                obs = env.step(a, on_substep=lambda txt=txt: frame(txt))
                box["img"] = obs
            for _ in range(int(0.6 * FPS)):
                frame([head, "aligned" if hold >= 5 else "not aligned"])
    env.close()
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
