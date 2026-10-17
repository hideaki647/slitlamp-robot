"""DAgger：生徒が自分で操作し、通った場面すべてに先生の正解を付けて教材に足し、学び直す。

1ラウンド = ①生徒で episodes 回操作（先生がラベル付け）→ ②全データで続きから学習 → ③未知の患者50人で評価。
BC だけだと、少しずれた「お手本に無い見え方」で生徒が戸惑う（分布シフト）。
DAgger は生徒が実際に行く場面を教材に加えるので、そのずれを直せる（Ross et al., 2011）。

実行（slitlamp_robot フォルダで、BC の学習後）:
    docker compose run --rm sim python learning/dagger.py --rounds 4
"""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
EVAL_SEED0 = 10_000          # scripts/eval_policy.py と同じ未知の患者条件
DAGGER_SEED0 = 20_000        # 先生のデータ（0〜）・評価（10000〜）と重ならない番号

_env, _policy, _ckpt = None, None, None


def _setup(ckpt):
    """各プロセスで環境を1つ作り、生徒の重みが変わったときだけ読み直す。"""
    global _env, _policy, _ckpt
    import torch
    from learning.model import StudentPolicy
    from sim.env import SlitLampEnv
    torch.set_num_threads(1)
    if _env is None:
        _env = SlitLampEnv()
    if _ckpt != ckpt:
        _policy, _ckpt = StudentPolicy(ckpt), ckpt


def _collect(seed, ckpt, out_dir, max_steps):
    from expert.expert_policy import expert_action
    from learning.rollout import run_episode
    _setup(ckpt)
    r = run_episode(_env, _policy, seed, max_steps=max_steps, hold_steps=max_steps + 1,
                    record=True, label_policy=expert_action)
    errs = r.pop("errors")
    np.savez_compressed(Path(out_dir) / f"ep_{seed:05d}.npz",
                        images=r.pop("images"), actions=r.pop("actions"), errors=errs)
    # 最後の5ステップがすべて 1mm 未満なら「合わせられた」とみなす（学習中の目安）
    return bool((np.linalg.norm(errs[-5:], axis=1) < 1.0).all())


def _evaluate(seed, ckpt):
    from learning.rollout import run_episode
    _setup(ckpt)
    return run_episode(_env, _policy, seed)


def evaluate(ex, ckpt, n=50):
    rs = list(ex.map(_evaluate, [EVAL_SEED0 + i for i in range(n)], [ckpt] * n))
    ok = [r for r in rs if r["success"]]
    return {"success_rate": len(ok) / n,
            "converge_time_s_mean": float(np.mean([r["converge_time_s"] for r in ok])) if ok else None,
            "final_error_mm_mean": float(np.mean([r["final_error_mm"] for r in rs]))}


def main():
    from learning.train_bc import train

    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--episodes", type=int, default=500, help="1ラウンドで生徒が操作する回数")
    p.add_argument("--max_steps", type=int, default=30, help="1エピソードの長さ（30＝3秒）")
    p.add_argument("--epochs", type=int, default=15, help="1ラウンドの学習エポック数（続きから学習）")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--start", default=str(CKPT / "bc.pt"))
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    data_dirs = [DATA / "demos"]
    ckpt = args.start
    progress_path = ROOT / "outputs" / "dagger_progress.json"
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        progress = [{"round": 0, "frames": 36000, **evaluate(ex, ckpt)}]
        print(progress[-1], flush=True)
        for rnd in range(1, args.rounds + 1):
            t0 = time.time()
            out_dir = DATA / "dagger" / f"r{rnd}"
            out_dir.mkdir(parents=True, exist_ok=True)
            seeds = [DAGGER_SEED0 + rnd * 1000 + i for i in range(args.episodes)]
            aligned = list(ex.map(_collect, seeds, [ckpt] * len(seeds), [str(out_dir)] * len(seeds),
                                  [args.max_steps] * len(seeds)))
            data_dirs.append(out_dir)

            new_ckpt = str(CKPT / f"dagger_r{rnd}.pt")
            train(data_dirs, new_ckpt, init=ckpt, epochs=args.epochs, lr=args.lr)
            ckpt = new_ckpt

            frames = 36000 + rnd * args.episodes * args.max_steps
            progress.append({"round": rnd, "frames": frames,
                             "collect_aligned_rate": float(np.mean(aligned)),
                             **evaluate(ex, ckpt), "seconds": round(time.time() - t0)})
            print(progress[-1], flush=True)
            progress_path.write_text(json.dumps(progress, indent=1))


if __name__ == "__main__":
    main()
