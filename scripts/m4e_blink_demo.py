"""M4-E 動画：瞬きの前後を、2つの方式で並べる（上：瞬きを学んでいない生徒、下：瞬きを学んだ生徒）。

同じ患者で、1.0秒と2.5秒に瞬きする（評価と同じ）。瞬きの間は、画面に「BLINK」と瞼の閉じ具合を表示する。
右下の黄色い枠＝生徒が見ている画像。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4e_blink_demo.py
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from scripts.eval_policy import EVAL_SEED0, load_policy
from scripts.m2_expert_demo import FPS, H, W, slit_view
from sim.env import CONTROL_DT, PUPIL_HIDDEN_AT, SlitLampEnv
from sim.viz import free_camera, label

ROOT = Path(__file__).resolve().parents[1]


def film(policy_name, title, seed, steps):
    """1つの方式で、steps ステップ分のコマ（左：俯瞰、右：細隙灯カメラ）を返す。"""
    policy = load_policy(policy_name)
    env = SlitLampEnv(blink="eval")
    m = env.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    view = free_camera([0.0, -0.05, 0.95], 1.2, 140, -25)
    fovy = m.cam_fovy[m.camera("slit_cam").id]
    frames, box = [], {}

    def frame(text):
        d = env.data
        dist = d.site_xpos[env.ids["pupil_site"]][1] - d.cam_xpos[env.ids["cam"]][1]
        renderer.update_scene(d, camera=view)
        blink = env.lid_closure > 0.05
        left = label(renderer.render(), text + ([f"BLINK  closure {env.lid_closure:.2f}"
                                                 + ("  (pupil hidden)" if env.lid_closure > PUPIL_HIDDEN_AT else "")]
                                                if blink else []))
        right = slit_view(renderer, d, env.error_mm(), fovy, dist, title="slit lamp camera  (inset: what the policy sees)")
        inset = cv2.resize(box["img"], (192, 192), interpolation=cv2.INTER_NEAREST)
        right[H - 200:H - 8, W - 200:W - 8] = inset
        cv2.rectangle(right, (W - 201, H - 201), (W - 7, H - 7), (255, 255, 0), 1)
        frames.append(np.hstack([left, right]))

    obs = env.reset(seed)
    box["img"] = obs
    for t in range(steps):
        a = policy(obs, env)
        txt = [title, f"t = {t * CONTROL_DT:.1f} s   command dx {a[0]:+4.1f}  dz {a[1]:+4.1f} mm"]
        obs = env.step(a, on_substep=lambda txt=txt: frame(txt))
        box["img"] = obs
    env.close()
    renderer.close()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", default="student:checkpoints/dagger_r4.pt")
    p.add_argument("--top_name", default="E2E student (BC+DAgger, never saw blinks)")
    p.add_argument("--bottom", default="student:checkpoints/bc_blink3.pt")
    p.add_argument("--bottom_name", default="E2E student trained with blinks (frame stack x3)")
    p.add_argument("--seed", type=int, default=EVAL_SEED0 + 2)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--out", default="outputs/m4e_blink_demo.mp4")
    args = p.parse_args()

    a = film(args.top, args.top_name, args.seed, args.steps)
    b = film(args.bottom, args.bottom_name, args.seed, args.steps)
    path = ROOT / args.out
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as w:
        for x, y in zip(a, b):
            w.append_data(cv2.resize(np.vstack([x, y]), (960, 720)))
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
