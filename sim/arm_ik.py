"""アームの逆運動学（IK）：手先(grip)を目標の位置・向きに置く関節角を求める。

方法：ヤコビアンを使った「減衰付き最小二乗法」(Damped Least Squares)
    1. 今の関節角で手先の位置・向きを計算する（順運動学）
    2. 目標とのずれ e（位置3つ＋向き3つ）を求める
    3. ヤコビアン J（関節を少し動かすと手先がどう動くかの表）を求める
    4. Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e だけ関節を動かす（λ が減衰。特異姿勢で暴れないようにする）
    5. ずれが十分小さくなるまで繰り返す
"""
import mujoco
import numpy as np

ARM_JOINTS = ["a_yaw", "a_shoulder", "a_elbow", "a_wrist_pitch", "a_wrist_dev", "a_wrist_roll"]


def rotation_error(current, target):
    """2つの回転行列の差を、回転ベクトル（軸×角度、3要素）で近似する。"""
    return 0.5 * sum(np.cross(current[:, i], target[:, i]) for i in range(3))


class ArmIK:
    def __init__(self, model, site="grip", damping=0.02):
        self.model = model
        self.data = mujoco.MjData(model)  # 計算専用のコピー（本番の data は動かさない）
        self.site = model.site(site).id
        self.damping = damping
        ids = [model.joint(j).id for j in ARM_JOINTS]
        self.qadr = model.jnt_qposadr[ids]
        self.dadr = model.jnt_dofadr[ids]
        self.lo, self.hi = model.jnt_range[ids].T  # 関節の可動範囲（ラジアン）

    def solve(self, target_pos, target_mat, q_init, iters=100, tol=1e-5, max_step=0.2):
        """目標の位置(3)・回転行列(3×3)に手先を合わせる関節角を返す。(関節角, 位置誤差[m], 角度誤差[rad])"""
        m, d = self.model, self.data
        q = np.clip(np.array(q_init, dtype=float), self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)  # 関節角 → 各部品の位置・向き
            mujoco.mj_comPos(m, d)      # ヤコビアン計算に必要な量を更新
            e_pos = target_pos - d.site_xpos[self.site]
            e_rot = rotation_error(d.site_xmat[self.site].reshape(3, 3), target_mat)
            e = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp, jacr])[:, self.dadr]  # アームの6関節の列だけ取り出す（6×6）
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
            dq = np.clip(dq, -max_step, max_step)  # 1回で大きく動きすぎないように
            q = np.clip(q + dq, self.lo, self.hi)
        return q, np.linalg.norm(e_pos), np.linalg.norm(e_rot)
