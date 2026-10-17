"""M3 デモ動画：生徒AI（細隙灯カメラの画像だけを見る）が細隙灯を瞳孔へ自動で合わせる。

M2 の先生の動画と同じ4人の患者条件・同じ画面構成で撮るので、そのまま見比べられる。
右の画面の左上に、生徒が実際に見ている 96×96 の画像を拡大して表示する。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m3_student_demo.py --ckpt checkpoints/dagger_r4.pt
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from learning.model import StudentPolicy
from scripts.m2_expert_demo import FPS, H, N_EPISODES, W, slit_view
from sim.env import CONTROL_DT, SUCCESS_MM, SlitLampEnv
from sim.viz import free_camera, label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/dagger_r4.pt")
    p.add_argument("--name", default="STUDENT (camera image only, BC+DAgger)")
    p.add_argument("--out", default="outputs/m3_student_demo.mp4")
    p.add_argument("--policy", default=None, help="eval_policy.py と同じ書き方（例：modular）。指定すると --ckpt より優先")
    p.add_argument("--eye", default="R", choices=["R", "L"])
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--delay", type=int, default=0, help="観測の遅れ [ステップ=0.1秒]（M4-G）")
    args = p.parse_args()

    evals = json.loads((OUT / "eval_expert.json").read_text())["episodes"]
    seeds = [e["seed"] for e in sorted(evals, key=lambda e: -e["initial_error_mm"])[:args.episodes]]

    if args.policy:
        from scripts.eval_policy import load_policy
        student = load_policy(args.policy)
    else:
        student = StudentPolicy(ROOT / args.ckpt)
    env = SlitLampEnv(target_eye=args.eye, disturb={"delay": args.delay})
    m = env.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    view = free_camera([0.0, -0.05, 0.95], 1.2, 140, -25)
    fovy = m.cam_fovy[m.camera("slit_cam").id]
    obs_box = {"img": None}

    path = ROOT / args.out
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:

        def frame(text):
            d = env.data
            dist = d.site_xpos[env.ids["pupil_site"]][1] - d.cam_xpos[env.ids["cam"]][1]
            renderer.update_scene(d, camera=view)
            left = label(renderer.render(), text)
            right = slit_view(renderer, d, env.error_mm(), fovy, dist,
                              title="slit lamp camera  (inset: what the student sees)")
            # 生徒の入力画像（直前に撮った 96×96）を右下に 2倍で貼る
            inset = cv2.resize(obs_box["img"], (192, 192), interpolation=cv2.INTER_NEAREST)
            right[H - 200:H - 8, W - 200:W - 8] = inset
            cv2.rectangle(right, (W - 201, H - 201), (W - 7, H - 7), (255, 255, 0), 1)
            writer.append_data(np.hstack([left, right]))

        for k, seed in enumerate(seeds):
            obs = env.reset(seed)
            obs_box["img"] = obs
            head = f"episode {k + 1}/{len(seeds)}  {args.name}"
            for _ in range(int(0.7 * FPS)):
                frame([head, "start"])
            hold = 0
            for t in range(50):
                hold = hold + 1 if np.linalg.norm(env.error_mm()) < SUCCESS_MM else 0
                if hold >= 5:
                    break
                a = student(obs)
                txt = [head, f"t = {t * CONTROL_DT:.1f} s", f"command dx {a[0]:+4.1f}  dz {a[1]:+4.1f} mm"]
                obs = env.step(a, on_substep=lambda txt=txt: frame(txt))
                obs_box["img"] = obs
            for _ in range(int(0.7 * FPS)):
                frame([head, "aligned" if hold >= 5 else "not aligned"])
    env.close()
    print("saved", path.relative_to(ROOT), "seeds", seeds)


if __name__ == "__main__":
    main()
