"""ポリシーの評価：学習に使っていない患者条件（seed 10000〜）で N 回試し、成功率・収束時間・最終誤差を集計する。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/eval_policy.py --policy expert
生徒は --policy student:<モデルのパス> --tag <結果ファイル名> で同じ条件で評価する:
    docker compose run --rm sim python scripts/eval_policy.py --policy student:checkpoints/bc.pt --tag bc
"""
import argparse
import json
from pathlib import Path

import numpy as np

from expert.expert_policy import expert_action
from learning.rollout import run_episode
from sim.env import CONDITIONS, SlitLampEnv

ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED0 = 10_000  # 学習データ（seed 0〜）と重ならない番号


def load_policy(name):
    if name == "expert":
        return expert_action
    if name.startswith("student:"):
        from learning.model import StudentPolicy
        return StudentPolicy(ROOT / name.split(":", 1)[1])
    raise ValueError(f"unknown policy: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="expert")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--tag", default=None, help="結果ファイル outputs/eval_<tag>.json の名前")
    p.add_argument("--condition", default="normal", choices=CONDITIONS, help="汎化テストの条件")
    p.add_argument("--eye", default="R", choices=["R", "L"], help="合わせる眼（患者の右眼 R／左眼 L）")
    args = p.parse_args()

    env = SlitLampEnv(condition=args.condition, target_eye=args.eye)
    policy = load_policy(args.policy)
    results = [run_episode(env, policy, EVAL_SEED0 + i) for i in range(args.n)]
    env.close()

    ok = [r for r in results if r["success"]]
    summary = {
        "policy": args.policy,
        "condition": args.condition,
        "eye": args.eye,
        "episodes": args.n,
        "success_rate": len(ok) / args.n,
        "converge_time_s_mean": float(np.mean([r["converge_time_s"] for r in ok])) if ok else None,
        "converge_time_s_max": float(np.max([r["converge_time_s"] for r in ok])) if ok else None,
        "final_error_mm_mean": float(np.mean([r["final_error_mm"] for r in results])),
        "initial_error_mm_mean": float(np.mean([r["initial_error_mm"] for r in results])),
    }
    for k, v in summary.items():
        print(f"{k:<24}{v:.3f}" if isinstance(v, float) else f"{k:<24}{v}")
    failed = [r["seed"] for r in results if not r["success"]]
    if failed:
        print("failed seeds:", failed)

    out = ROOT / "outputs" / f"eval_{args.tag or args.policy.split(':')[0]}.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=float))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
