"""M6-②：倍率ごとの見え方の確認。

1. 一覧画像 outputs/m6_mag_check.png：同じ患者で、倍率（行）× 焦点からのずれ（列）の細隙灯カメラの画像（生徒が見る 96×96）。
2. 1mm が何画素か：本体を左右に 1mm 動かしたとき、検出した瞳孔が画像の上で何画素動くかを、倍率ごとに測る。
3. 開始位置で瞳孔が視野に入っている割合（未知の患者50人、これまでと同じ開始位置 左右±15mm・上下±8mm）。
4. 先生（特権情報）が、どの倍率でも合わせられるか（10人ずつ）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m6_mag_check.py
"""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from expert.expert_policy import expert_action
from learning.rollout import run_episode
from policies.modular import detect_pupil
from sim.env import FIELD_MM, MAGS, SlitLampEnv

ROOT = Path(__file__).resolve().parents[1]
SEED = 10001
FOCUS_MM = [-1.0, 0.0, 0.2, 0.5, 1.0, 3.0]   # 焦点からのずれ。− は近づきすぎ、＋ は手前すぎ
SCALE = 2


def aligned(env):
    """リセット直後の env で、光軸が瞳孔中心を通り、ピントも合う本体の位置 (x, z, y) [m]。"""
    e = env.error_mm()
    return env.target + e[:2] / 1000, env.target_y + e[2] / 1000


def grid():
    rows = []
    for mag in MAGS:
        env = SlitLampEnv(lamp="sl130", mag=mag, focus="back")
        env.reset(SEED)
        xz, y0 = aligned(env)
        tiles = []
        for f in FOCUS_MM:
            env.robot.grasp(*xz, y=y0 - f / 1000)        # 本体を焦点から f mm 手前へ（− なら奥へ）
            img = cv2.resize(env.observe(), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            label = f"x{mag}  {'focus' if f == 0 else f'{f:+.1f}mm'}"
            cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
        env.close()
    imageio.imwrite(ROOT / "outputs" / "m6_mag_check.png", np.vstack(rows))
    print("saved outputs/m6_mag_check.png")


def px_per_mm():
    print("\n1mm が何画素か（本体を左右に ±1mm 動かしたときの、検出した瞳孔の動き）")
    for mag in MAGS:
        env = SlitLampEnv(lamp="sl130", mag=mag)
        env.reset(SEED)
        xz = env.target + env.error_mm() / 1000
        cx = []
        for dx in (-0.001, 0.001):
            env.robot.grasp(xz[0] + dx, xz[1])
            found = detect_pupil(env.observe(), px_per_mm=env.px_per_mm)
            cx.append(found[0] if found is not None else np.nan)
        print(f"  x{mag:<3d} 視野 {FIELD_MM[mag]:4.0f}mm  理論 {env.px_per_mm:5.2f}  測定 {abs(cx[1] - cx[0]) / 2:5.2f} 画素/mm")
        env.close()


def start_in_view(n=50):
    print("\n開始位置で瞳孔の中心が視野（丸）に入っている割合（未知の患者50人）")
    env = SlitLampEnv()
    offsets = []
    for seed in range(10000, 10000 + n):
        env.reset(seed)
        offsets.append(np.linalg.norm(env.error_mm()))
    env.close()
    offsets = np.array(offsets)
    for mag in MAGS:
        print(f"  x{mag:<3d} 視野の半径 {FIELD_MM[mag] / 2:5.1f}mm  入っている {np.mean(offsets < FIELD_MM[mag] / 2) * 100:5.0f}%")


def expert(n=10):
    print("\n先生（特権情報）：倍率を固定して10人ずつ")
    for mag in MAGS:
        env = SlitLampEnv(lamp="sl130", mag=mag)
        rs = [run_episode(env, expert_action, seed) for seed in range(10000, 10000 + n)]
        print(f"  x{mag:<3d} 成功 {np.mean([r['success'] for r in rs]) * 100:5.0f}%  最終誤差 {np.mean([r['final_error_mm'] for r in rs]):.3f}mm")
        env.close()


if __name__ == "__main__":
    grid()
    px_per_mm()
    start_in_view()
    expert()
