"""M4-H 動画：ピント合わせ。同じ患者で、2つの方式（または2つの開始位置）を上下に並べる。

右の細隙灯カメラの画像に、前後のずれ（焦点まで何mm）を表示する。右下の黄色い枠＝生徒が見ている画像（ボケあり）。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4h_focus_demo.py
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from scripts.eval_policy import EVAL_SEED0, load_policy
from scripts.m2_expert_demo import FPS, H, W, slit_view
from sim.env import CONTROL_DT, FOCUS_TOL_MM, SlitLampEnv
from sim.viz import free_camera, label

ROOT = Path(__file__).resolve().parents[1]


def film(policy_name, title, focus, seed, steps):
    policy = load_policy(policy_name)
    env = SlitLampEnv(focus=focus)
    m = env.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    view = free_camera([0.0, -0.08, 0.95], 0.9, 180, -15)    # 横から：前後の動きが見える
    fovy = m.cam_fovy[m.camera("slit_cam").id]
    frames, box = [], {}

    def frame(text):
        d = env.data
        e = env.error_mm()
        dist = d.site_xpos[env.ids["pupil_site"]][1] - d.cam_xpos[env.ids["cam"]][1]
        renderer.update_scene(d, camera=view)
        state = ("IN FOCUS" if abs(e[2]) < FOCUS_TOL_MM
                 else f"{abs(e[2]):.1f} mm {'too far (push in)' if e[2] > 0 else 'too close (pull back)'}")
        left = label(renderer.render(), text + [f"focus: {state}"])
        right = slit_view(renderer, d, e[:2], fovy, dist, title="slit lamp camera (no blur)  inset: what the policy sees")
        inset = cv2.resize(box["img"], (192, 192), interpolation=cv2.INTER_NEAREST)
        right[H - 200:H - 8, W - 200:W - 8] = inset
        cv2.rectangle(right, (W - 201, H - 201), (W - 7, H - 7), (255, 255, 0), 1)
        frames.append(np.hstack([left, right]))

    obs = env.reset(seed)
    box["img"] = obs
    hold = 0
    for t in range(steps):
        hold = hold + 1 if env.is_aligned() else 0
        if hold >= 5:
            break
        a = policy(obs, env)
        txt = [title, f"t = {t * CONTROL_DT:.1f} s  cmd dx {a[0]:+4.1f} dz {a[1]:+4.1f} dy {a[2]:+4.1f} mm"]
        obs = env.step(a, on_substep=lambda txt=txt: frame(txt))
        box["img"] = obs
    for _ in range(int(0.6 * FPS)):
        frame([title, "aligned" if hold >= 5 else "not aligned"])
    env.close()
    renderer.close()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", default="student:checkpoints/bc_focus3.pt")
    p.add_argument("--top_name", default="student (3-frame stack), start: pulled back (usual)")
    p.add_argument("--top_focus", default="back")
    p.add_argument("--bottom", default="student:checkpoints/bc_focus1.pt")
    p.add_argument("--bottom_name", default="student (1 frame), start: TOO CLOSE")
    p.add_argument("--bottom_focus", default="front")
    p.add_argument("--seed", type=int, default=EVAL_SEED0 + 2)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="outputs/m4h_focus_demo.mp4")
    args = p.parse_args()

    a = film(args.top, args.top_name, args.top_focus, args.seed, args.steps)
    b = film(args.bottom, args.bottom_name, args.bottom_focus, args.seed, args.steps)
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))
    path = ROOT / args.out
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as w:
        for x, y in zip(a, b):
            w.append_data(cv2.resize(np.vstack([x, y]), (960, 720)))
    print("saved", path.relative_to(ROOT), f"{n / FPS:.1f} s")


if __name__ == "__main__":
    main()
