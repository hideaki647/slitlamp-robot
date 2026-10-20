"""先生（エキスパート）：特権情報（瞳孔中心の真の位置）を使う P制御。

    指令 = ゲイン × ずれ    （1ステップ＝0.1秒ごと）

ゲイン 0.5 なら、毎ステップ ずれの半分だけ動かす → ずれは 1/2, 1/4, 1/8 … と縮む。
画像は見ないので、実機では使えない。生徒（M3）はこの指令を画像だけから真似る。
"""
import numpy as np

from sim.env import MAX_STEP_MM

GAIN = 0.5


def expert_action(obs, env):
    """obs（画像）は使わず、env の特権情報からずれを読んで指令 (Δx, Δz)[mm] を返す。
    観測の遅れ（M4-G）があるときは、生徒と同じく遅れたずれを使う（遅れが無ければ今のずれと同じ）。
    瞬き（M4-E）で瞳孔が半分以上隠れている間は待つ（指令0）。先生は本当の位置を知っているので動けてしまうが、
    生徒は画像から同じことができないので、生徒が真似できるふるまいにそろえる（臨床でも瞬きの間は待つ）。"""
    if getattr(env, "pupil_hidden", False):
        return np.zeros(2)
    return np.clip(GAIN * env.observed_error_mm(), -MAX_STEP_MM, MAX_STEP_MM)
