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
    if name in ("modular", "modular_raw"):
        from policies.modular import ModularPolicy
        return ModularPolicy(calibrate=(name == "modular"))
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
    # 実機の不完全さ（M4-G）。既定はすべて「なし」
    p.add_argument("--delay", type=int, default=0, help="観測の遅れ [ステップ=0.1秒]")
    p.add_argument("--act_noise", type=float, default=0.0, help="指令のばらつき [mm]")
    p.add_argument("--kp_scale", type=float, default=1.0, help="モーターの強さの倍率")
    p.add_argument("--img_noise", type=float, default=0.0, help="画像のノイズ（0〜1）")
    p.add_argument("--blur", type=float, default=0.0, help="画像のぼけ [画素]")
    p.add_argument("--out_scale", type=float, default=1.0,
                   help="方式の指令に掛ける倍率（M4-G 案3：遅れに合わせてゲインを下げる。例 0.15/0.5=0.3）")
    args = p.parse_args()

    disturb = {k: getattr(args, k) for k in ("delay", "act_noise", "kp_scale", "img_noise", "blur")}
    env = SlitLampEnv(condition=args.condition, target_eye=args.eye, disturb=disturb)
    policy = load_policy(args.policy)
    run_policy = policy
    if args.out_scale != 1.0:  # 指令に倍率を掛ける（学習し直しなしでゲインを下げる）
        def run_policy(obs, env, _p=policy, _s=args.out_scale):
            return _s * np.asarray(_p(obs, env))
    results = [run_episode(env, run_policy, EVAL_SEED0 + i) for i in range(args.n)]
    env.close()

    ok = [r for r in results if r["success"]]
    summary = {
        "policy": args.policy,
        "condition": args.condition,
        "eye": args.eye,
        "disturb": disturb,
        "out_scale": args.out_scale,
        "episodes": args.n,
        "success_rate": len(ok) / args.n,
        "converge_time_s_mean": float(np.mean([r["converge_time_s"] for r in ok])) if ok else None,
        "converge_time_s_max": float(np.max([r["converge_time_s"] for r in ok])) if ok else None,
        "final_error_mm_mean": float(np.mean([r["final_error_mm"] for r in results])),
        "initial_error_mm_mean": float(np.mean([r["initial_error_mm"] for r in results])),
        # 合った後の最大のずれ（一度でも1mm以内に入った回の平均・最大）、振動の回数の平均（M4-G）
        "post_align_max_mm_mean": float(np.mean(pm)) if (pm := [r["post_align_max_mm"] for r in results
                                                               if r["post_align_max_mm"] is not None]) else None,
        "post_align_max_mm_max": float(np.max(pm)) if pm else None,
        "oscillations_mean": float(np.mean([r["oscillations"] for r in results])),
    }
    if hasattr(policy, "misses"):  # モジュール型：瞳孔を見つけられなかった割合
        summary["detect_miss_rate"] = policy.misses / max(policy.calls, 1)
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
