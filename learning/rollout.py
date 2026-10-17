"""1エピソード（1回の診察）を走らせる共通の関数。先生・生徒の両方で使う。"""
import numpy as np

from sim.env import CONTROL_DT, SUCCESS_MM


def run_episode(env, policy, seed, max_steps=50, hold_steps=5, noise_std_mm=0.0,
                record=False, on_substep=None, label_policy=None):
    """
    policy(obs, env) → 指令 (Δx, Δz)[mm]。
    noise_std_mm > 0 なら、実際に動かす指令にだけノイズを混ぜる（DART）。記録するのはノイズなしの指令。
    label_policy を渡すと、動かすのは policy、記録する正解ラベルは label_policy の指令になる（DAgger）。
    ずれ < SUCCESS_MM が hold_steps 回続いたら成功として終了。
    """
    rng = np.random.default_rng(seed + 1_000_000)
    obs = env.reset(seed)
    initial = float(np.linalg.norm(env.error_mm()))
    images, actions, errors = [], [], []
    converged_at, hold = None, 0

    for t in range(max_steps):
        err = env.error_mm()
        if np.linalg.norm(err) < SUCCESS_MM:
            hold += 1
            if converged_at is None:
                converged_at = t
            if hold >= hold_steps:
                break
        else:
            hold, converged_at = 0, None

        action = np.asarray(policy(obs, env), dtype=float)
        if record:
            images.append(obs)
            actions.append(action if label_policy is None else label_policy(obs, env))
            errors.append(err)
        executed = action + rng.normal(0, noise_std_mm, 2) if noise_std_mm > 0 else action
        obs = env.step(executed, on_substep=on_substep)

    final = env.error_mm()
    min_err = min(np.linalg.norm(e) for e in errors) if record and errors else None
    result = {
        "min_error_mm": min_err,
        "seed": seed,
        "success": hold >= hold_steps,
        "converge_time_s": converged_at * CONTROL_DT if hold >= hold_steps else None,
        "final_error_mm": float(np.linalg.norm(final)),
        "initial_error_mm": initial,
        "steps": env.steps,
        **env.info,
    }
    if record:
        result["images"] = np.array(images, dtype=np.uint8)
        result["actions"] = np.array(actions, dtype=np.float32)
        result["errors"] = np.array(errors, dtype=np.float32)
    return result
