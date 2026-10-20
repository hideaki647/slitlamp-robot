"""先生のお手本（デモ）を集める。1エピソード＝1ファイル data/demos/ep_XXXXX.npz。

各ファイルの中身:
    images  (T, 96, 96, 3) uint8   … 細隙灯カメラの画像（生徒の入力）
    actions (T, 2) float32 [mm]     … そのとき先生が出した指令 (Δx, Δz)（生徒の正解ラベル）
    errors  (T, 2) float32 [mm]     … 瞳孔中心と光軸のずれ（分析用。生徒には見せない）

DART：実際に動かす指令にだけノイズ（標準偏差 noise_std mm）を混ぜ、ずれた状態から戻る場面も集める。
各エピソードは max_steps（既定12＝1.2秒）で打ち切る。先生は約0.5秒で合わせるので、
「近づく場面」と「合った後の微調整の場面」がおよそ半々になる。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python learning/collect_demos.py --episodes 3000
"""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demos"

_env = None  # 各プロセスで1つだけ作る環境


def _worker(seed, noise_std, max_steps, eye, out_dir, switch=False, blink=False):
    global _env
    from expert.expert_policy import expert_action
    from learning.rollout import run_episode
    from sim.env import SlitLampEnv

    if _env is None:
        # 両眼切替（M4-F）：目標の眼はランダム、反対の眼に合わせた状態から始め、両眼の間の距離もばらつかせる
        _env = (SlitLampEnv(eye_mode="both", start="switch", ipd_mm=(56, 72)) if switch
                else SlitLampEnv(blink="random") if blink        # 瞬きあり（M4-E）：右眼、ランダムな時刻に瞬き
                else SlitLampEnv(target_eye=eye))
    # hold_steps を max_steps より大きくして、途中で打ち切らず毎回 max_steps 分を記録する
    r = run_episode(_env, expert_action, seed, max_steps=max_steps, hold_steps=max_steps + 1,
                    noise_std_mm=noise_std, record=True)
    np.savez_compressed(Path(out_dir) / f"ep_{seed:05d}.npz",
                        images=r.pop("images"), actions=r.pop("actions"), errors=r.pop("errors"), goal=_env.goal)
    return r


def sample_grid(out_dir=OUT_DIR, name="m2_demo_samples.png", n=48, cols=8, scale=2):
    """データの中身を目で確認するための一覧画像。矢印＝先生の指令の向きと大きさ。"""
    rng = np.random.default_rng(0)
    files = sorted(Path(out_dir).glob("ep_*.npz"))
    tiles = []
    for f in rng.choice(files, n):
        d = np.load(f)
        t = rng.integers(len(d["actions"]))
        img = cv2.resize(d["images"][t], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        c = img.shape[0] // 2
        ax, az = d["actions"][t]
        # 画像の右＝+x、上＝+z。1mm を 8画素で表示
        tip = (int(c + ax * 8), int(c - az * 8))
        cv2.drawMarker(img, (c, c), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.arrowedLine(img, (c, c), tip, (255, 0, 0), 2, tipLength=0.3)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, n, cols)]
    path = ROOT / "outputs" / name
    imageio.imwrite(path, np.vstack(rows))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--noise_std", type=float, default=0.5, help="DART のノイズ [mm]")
    p.add_argument("--max_steps", type=int, default=12)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eye", default="R", choices=["R", "L"], help="合わせる眼（左眼なら data/demos_L に保存）")
    p.add_argument("--blink", action="store_true",
                   help="瞬きありのお手本（M4-E）。data/demos_blink に保存。--max_steps 30 を推奨")
    p.add_argument("--switch", action="store_true",
                   help="両眼切替のお手本（M4-F）。data/demos_switch に保存。--max_steps 40 を推奨")
    args = p.parse_args()
    out_dir = (ROOT / "data" / "demos_switch" if args.switch
               else ROOT / "data" / "demos_blink" if args.blink
               else OUT_DIR if args.eye == "R" else ROOT / "data" / "demos_L")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    n = args.episodes
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        results = list(ex.map(_worker, range(n), [args.noise_std] * n, [args.max_steps] * n,
                              [args.eye] * n, [str(out_dir)] * n, [args.switch] * n, [args.blink] * n))
    dt = time.time() - t0

    frames = sum(r["steps"] for r in results)
    summary = {
        "episodes": args.episodes,
        "frames": frames,
        "noise_std_mm": args.noise_std,
        "max_steps": args.max_steps,
        "eye": args.eye,
        "switch": args.switch,
        "blink": args.blink,
        "reached_1mm_rate": float(np.mean([r["min_error_mm"] < 1.0 for r in results])),
        "seconds": round(dt, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=float))
    print(summary)
    grid = ("m4f_demo_samples_switch.png" if args.switch
            else "m4e_demo_samples_blink.png" if args.blink
            else "m2_demo_samples.png" if args.eye == "R" else "m3_demo_samples_L.png")
    print("saved", sample_grid(out_dir, grid).relative_to(ROOT))


if __name__ == "__main__":
    main()
