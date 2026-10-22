"""細隙灯の操作API：「本体を x, z へ」→ ジョイスティック角 → ノブの目標姿勢 → IK → アームのモーター指令。

M3 で学習するポリシー（生徒）は、このAPIの command() に渡す値を出す役になる。
"""
import mujoco
import numpy as np

from sim.arm_ik import ARM_JOINTS, ArmIK

# 連動拘束の係数（scene.xml の polycoef と同じ値）
K_XY = 0.15     # 傾き1 rad あたりの本体移動 [m]
C_Z = 0.00318   # ハンドル回転1 rad あたりの高さ変化 [m]（半回転で約10mm）
ROLL_MAX = np.deg2rad(24)    # ジョイスティックの可動範囲（25°）の少し内側
SPIN_MAX = np.deg2rad(350)
X_MAX = K_XY * ROLL_MAX      # 本体が動ける左右の範囲 ±約63mm
Z_MAX = C_Z * SPIN_MAX       # 本体が動ける高さの範囲 ±約19mm
Y_MAX = 0.035                # 本体が動ける前後の範囲 ±35mm（lamp_y の可動範囲 ±40mm の内側。M4-H のピント合わせ）

# 握ったときの手とノブの向きの関係：手は下向き、ノブの軸は上向き → x軸まわりに180°反転
GRIP_FLIP = np.diag([1.0, -1.0, -1.0])
# IKの初期値（肩を患者側へ倒し、肘を曲げ、手首を下に向けたおおよその姿勢）[rad]
Q_GUESS = np.deg2rad([0, -30, -100, -50, 0, 0])
WRIST = ARM_JOINTS.index("a_wrist_roll")


class SlitLampRobot:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ik = ArmIK(model)
        self.calc = mujoco.MjData(model)  # ノブの目標姿勢を計算する専用のコピー
        self.q = {n: model.joint(n).qposadr[0] for n in
                  ("lamp_x", "lamp_y", "lamp_z", "js_roll", "js_pitch", "js_spin")}
        self.arm_qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.act = np.array([model.actuator("m_" + j[2:]).id for j in ARM_JOINTS])
        self.knob_site = model.site("js_grip").id
        self.weld = model.equality("grasp").id
        self.q_arm = Q_GUESS.copy()

    # ---------- 換算 ----------
    def lamp_to_joystick(self, x, z, y=0.0):
        """本体の目標位置 (x, z, y)[m] → ジョイスティック角 (roll, spin, pitch)[rad]。可動範囲で打ち切る。
        前後 y は、ジョイスティックを前後に倒して動かす（lamp_y = −K_XY × pitch、M4-H のピント合わせ）。"""
        roll = np.clip(x / K_XY, -ROLL_MAX, ROLL_MAX)
        spin = np.clip(z / C_Z, -SPIN_MAX, SPIN_MAX)
        pitch = -np.clip(y, -Y_MAX, Y_MAX) / K_XY
        return roll, spin, pitch

    def knob_pose(self, roll, spin, pitch=0.0):
        """ジョイスティックを (roll, spin, pitch) にしたときのノブの位置と向き。
        ジョイスティックは基台に付いているので、本体が動く分も含めて計算する。"""
        c, q = self.calc, self.q
        c.qpos[:] = self.model.qpos0
        c.qpos[q["js_roll"]], c.qpos[q["js_spin"]], c.qpos[q["js_pitch"]] = roll, spin, pitch
        c.qpos[q["lamp_x"]], c.qpos[q["lamp_z"]] = K_XY * roll, C_Z * spin  # 連動拘束と同じ関係
        c.qpos[q["lamp_y"]] = -K_XY * pitch
        mujoco.mj_kinematics(self.model, c)
        return c.site_xpos[self.knob_site].copy(), c.site_xmat[self.knob_site].reshape(3, 3).copy()

    def arm_target(self, roll, spin, pitch=0.0, iters=30):
        """ノブを (roll, spin, pitch) にするためのアーム関節角を IK で求める（前回の解から出発）。"""
        pos, mat = self.knob_pose(roll, spin, pitch)
        self.q_arm, e_pos, e_rot = self.ik.solve(pos, mat @ GRIP_FLIP, self.q_arm, iters=iters)
        return self.q_arm, e_pos, e_rot

    # ---------- 操作 ----------
    def grasp(self, x=0.0, z=0.0, y=0.0):
        """本体を (x, z, y)[m] に置いた状態で、アームをノブの位置へ直接置き、
        weld を ON にして「握った状態」にする（把持動作は省略）。"""
        m, d = self.model, self.data
        roll, spin, pitch = self.lamp_to_joystick(x, z, y)
        d.qpos[self.q["js_roll"]], d.qpos[self.q["js_spin"]], d.qpos[self.q["js_pitch"]] = roll, spin, pitch
        d.qpos[self.q["lamp_x"]], d.qpos[self.q["lamp_z"]] = K_XY * roll, C_Z * spin
        d.qpos[self.q["lamp_y"]] = -K_XY * pitch
        # 手首の回転は「ハンドル角の逆向き＋一定のずれ」で決まるが、向きだけを見る IK には
        # 360°ずれた解も同じに見える。ハンドル0°の姿勢から連続につながる解を選ばないと、
        # 手首が可動範囲の端に寄り、少し回しただけで限界に当たる（M3：左眼で発見）。
        self.q_arm = Q_GUESS.copy()
        self.arm_target(roll, 0.0, pitch, iters=300)
        self.q_arm[WRIST] -= spin
        q, e_pos, e_rot = self.arm_target(roll, spin, pitch, iters=300)
        d.qpos[self.arm_qadr] = q
        d.ctrl[self.act] = q
        mujoco.mj_forward(m, d)

        # weld の相対姿勢 relpose = 手(body1)から見たジョイスティック(body2)の位置・向き
        b1, b2 = m.body("hand").id, m.body("joystick").id
        rel_pos = d.xmat[b1].reshape(3, 3).T @ (d.xpos[b2] - d.xpos[b1])
        q1_inv = np.zeros(4)
        rel_quat = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, d.xquat[b1])
        mujoco.mju_mulQuat(rel_quat, q1_inv, d.xquat[b2])
        m.eq_data[self.weld, 3:6] = rel_pos
        m.eq_data[self.weld, 6:10] = rel_quat
        d.eq_active[self.weld] = 1
        return e_pos, e_rot

    def command(self, x, z, y=0.0):
        """本体を (x, z, y)[m] へ動かすように、アームのモーター目標を更新する。"""
        roll, spin, pitch = self.lamp_to_joystick(x, z, y)
        q, e_pos, e_rot = self.arm_target(roll, spin, pitch)
        self.data.ctrl[self.act] = q
        return e_pos, e_rot

    def lamp_position(self):
        """本体の現在位置 (x, z)[m]。"""
        return self.data.qpos[self.q["lamp_x"]], self.data.qpos[self.q["lamp_z"]]

    def weld_error(self):
        """手とノブのずれ [m]（握れていれば ほぼ 0）。"""
        d = self.data
        return np.linalg.norm(d.site_xpos[self.model.site("grip").id] - d.site_xpos[self.knob_site])
