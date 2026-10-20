"""1エピソード（1回の診察）を走らせる共通の関数。先生・生徒の両方で使う。"""
import numpy as np

from sim.env import CONTROL_DT, PUPIL_HIDDEN_AT, SUCCESS_MM


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
    path = []   # 各ステップのずれ（振動・行き過ぎの計算用、M4-G）
    lamp, closures = [], []   # 各ステップの本体の目標位置 [mm] と瞼の閉じ具合（瞬きの指標用、M4-E）

    for t in range(max_steps):
        err = env.error_mm()
        path.append(err)
        lamp.append(env.target * 1000)
        closures.append(env.lid_closure)
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
    path.append(final)
    lamp.append(env.target * 1000)
    closures.append(env.lid_closure)
    min_err = min(np.linalg.norm(e) for e in errors) if record and errors else None
    # 合った後の最大のずれ：一度 1mm 以内に入ったあと、どこまで離れたか（行き過ぎ・ふらつき、安全性の目安）
    norms = np.linalg.norm(np.array(path), axis=1)
    inside = np.nonzero(norms < SUCCESS_MM)[0]
    post_max = float(norms[inside[0] + 1:].max()) if len(inside) and inside[0] + 1 < len(norms) else None
    # 振動の回数：左右のずれの向きが入れ替わった回数（0.2mm 未満の小さな揺れは数えない）
    xs = [p[0] for p in path if abs(p[0]) >= 0.2]
    oscillations = int(sum(np.sign(a) != np.sign(b) for a, b in zip(xs, xs[1:])))
    result = {
        # 両眼切替（M4-F）：目標の1mm以内に初めて入った時刻、最初の3ステップで目標から遠ざかったか
        "first_in_s": float(inside[0] * CONTROL_DT) if len(inside) else None,
        "away_first3": bool(len(norms) > 3 and norms[3] > norms[0]),
        "post_align_max_mm": post_max,
        "oscillations": oscillations,
        "min_error_mm": min_err,
        "seed": seed,
        "success": hold >= hold_steps,
        "converge_time_s": converged_at * CONTROL_DT if hold >= hold_steps else None,
        "final_error_mm": float(np.linalg.norm(final)),
        "initial_error_mm": initial,
        "steps": env.steps,
        **env.info,
    }
    if getattr(env, "blink", None) in ("random", "eval"):
        result.update(blink_metrics(norms, np.array(lamp), np.array(closures)))
    if record:
        result["images"] = np.array(images, dtype=np.uint8)
        result["actions"] = np.array(actions, dtype=np.float32)
        result["errors"] = np.array(errors, dtype=np.float32)
    return result


def blink_metrics(norms, lamp, closures):
    """瞬きの指標（M4-E）。norms：各ステップのずれ [mm]、lamp：本体の目標位置 [mm]、closures：瞼の閉じ具合。
    瞳孔が半分以上隠れているステップ（閉じ具合 > PUPIL_HIDDEN_AT）を「瞬き中」とする。"""
    hidden = closures > PUPIL_HIDDEN_AT
    moves, recoveries = [], []
    i = 0
    while i < len(hidden):
        if not hidden[i]:
            i += 1
            continue
        j = i
        while j < len(hidden) and hidden[j]:
            j += 1
        before = max(i - 1, 0)                       # 瞬きの直前
        # 瞬き中の最大の動き：直前の本体の位置から、瞬き中（開いた直後まで）にどれだけ動いたか
        moves.append(float(np.max(np.linalg.norm(lamp[before:min(j + 1, len(lamp))] - lamp[before], axis=1))))
        # 合わせ直しの時間：瞬きの直前に合っていた場合だけ、開いてから 1mm 以内に戻るまで
        if norms[before] < SUCCESS_MM:
            back = np.nonzero(norms[j:] < SUCCESS_MM)[0]
            recoveries.append(float(back[0] * CONTROL_DT) if len(back) else None)
        i = j
    inside = np.nonzero(norms < SUCCESS_MM)[0]
    if len(inside):
        visible = ~hidden[inside[0]:]
        aligned_frac = float(np.mean(norms[inside[0]:][visible] < SUCCESS_MM)) if visible.any() else None
    else:
        aligned_frac = 0.0
    return {
        "blink_move_max_mm": max(moves) if moves else None,
        "blink_recover_s": [r for r in recoveries],
        "aligned_fraction": aligned_frac,
        "final_aligned": bool(np.all(norms[-5:] < SUCCESS_MM)),
    }
