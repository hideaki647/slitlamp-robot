"""両眼切替の比較用の方式（M4-F）：「決め打ちの移動」＋「モジュール型」。学習は使わない。

① 指示の向き（左眼なら検者から見て右＝＋x、右眼なら−x）へ、標準の瞳孔間距離 64mm だけ、1回最大3mmで進む
② 進み終わったら、モジュール型（M4-A：瞳孔検出＋比例制御）で合わせる
従来型のロボットの作り方の代表。患者の瞳孔間距離が 64mm からずれていても、移動後に目標の眼が
視野（横 約±26mm）に入っていれば、②で合わせ直せる。
"""
import numpy as np

from policies.modular import ModularPolicy
from sim.env import MAX_STEP_MM

NOMINAL_IPD_MM = 64.0


class SwitchBaseline:
    def __init__(self):
        self.modular = ModularPolicy()
        self.moved = 0.0
        self.last_steps = None

    def __call__(self, obs, env):
        if env.steps == 0 and self.last_steps != 0:   # 新しい患者：移動量を数え直す
            self.moved = 0.0
        self.last_steps = env.steps
        direction = 1.0 if env.goal == 1 else -1.0      # 指示だけを使う（目標の位置は見ない）
        remaining = NOMINAL_IPD_MM - self.moved
        if remaining > 1e-6:
            step = min(MAX_STEP_MM, remaining)
            self.moved += step
            return np.array([direction * step, 0.0])
        return self.modular(obs)
