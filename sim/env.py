"""細隙灯アライメント環境：リセット（ランダム化）・1ステップ進める・撮影・ずれの計算。

1ステップ = 0.1秒。行動（action）は「本体をあとどれだけ動かすか」の差分 (Δx, Δz) [mm]。
M2 の先生、M3 の生徒（学習ポリシー）の両方がこの環境を使う。
"""
from pathlib import Path

import mujoco
import numpy as np

from sim.slitlamp_api import X_MAX, Z_MAX, SlitLampRobot

ROOT = Path(__file__).resolve().parents[1]

CONTROL_DT = 0.1      # 1ステップの長さ [s]（10 Hz）
SUBSTEPS = 5          # 1ステップの中で指令をなめらかに5分割する
MAX_STEP_MM = 3.0     # 1ステップで動かせる最大量 [mm]
SUCCESS_MM = 1.0      # 瞳孔中心と光軸のずれがこれ未満なら「合っている」

# 虹彩の色の候補（RGB）：濃い茶・茶・ヘーゼル・緑・青
IRIS_COLORS = np.array([
    [0.25, 0.15, 0.08], [0.40, 0.25, 0.12], [0.50, 0.38, 0.18],
    [0.35, 0.45, 0.25], [0.30, 0.45, 0.65],
])


# 汎化テスト用の「学習時に無かった条件」（M3）
CONDITIONS = ["normal", "dark", "bright", "gray_iris", "head_far"]


class SlitLampEnv:
    def __init__(self, img_size=96, seed=None, target_eye="R", condition="normal"):
        assert condition in CONDITIONS, condition
        self.condition = condition
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.robot = SlitLampRobot(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=img_size, width=img_size)
        self.rng = np.random.default_rng(seed)
        m = self.model

        self.ids = {
            "head": m.body("head").id,
            "pupil_site": m.site(f"pupil_{target_eye}_center").id,
            "cam": m.camera("slit_cam").id,
            "pupils": [m.geom("pupil_R").id, m.geom("pupil_L").id],
            "iris_mat": m.material("iris_mat").id,
            "skin_mat": m.material("skin_mat").id,
            "slit_light": m.light("slit_light").id,
        }
        # ランダム化の基準値（毎回ここから変える）
        self.base = {
            "head_pos": m.body_pos[self.ids["head"]].copy(),
            "skin": m.mat_rgba[self.ids["skin_mat"]].copy(),
            "slit_diffuse": m.light_diffuse[self.ids["slit_light"]].copy(),
            "head_diffuse": m.vis.headlight.diffuse.copy(),
            "head_ambient": m.vis.headlight.ambient.copy(),
        }
        self.substep_n = round(CONTROL_DT / SUBSTEPS / m.opt.timestep)  # 1分割あたりの物理ステップ数
        self.target = np.zeros(2)  # 本体の目標位置 (x, z) [m]
        self.info = {}

    # ---------- リセット ----------
    def _randomize(self):
        m, r, ids, b = self.model, self.rng, self.ids, self.base
        head_dx, head_dz = r.uniform(-0.010, 0.010), r.uniform(-0.008, 0.008)
        m.body_pos[ids["head"]] = b["head_pos"] + [head_dx, 0, head_dz]   # 顎台での座り方の違い
        pupil_r = r.uniform(0.0015, 0.004)                                    # 縮瞳〜散瞳
        for g in ids["pupils"]:
            m.geom_size[g][0] = pupil_r
        iris = IRIS_COLORS[r.integers(len(IRIS_COLORS))] * r.uniform(0.85, 1.15, 3)
        m.mat_rgba[ids["iris_mat"], :3] = np.clip(iris, 0, 1)
        m.mat_rgba[ids["skin_mat"], :3] = np.clip(b["skin"][:3] + r.uniform(-0.06, 0.06, 3), 0, 1)
        m.light_diffuse[ids["slit_light"]] = b["slit_diffuse"] * r.uniform(0.7, 1.3)
        s = r.uniform(0.7, 1.3)
        m.vis.headlight.diffuse = b["head_diffuse"] * s
        m.vis.headlight.ambient = b["head_ambient"] * s

        # 汎化テスト：学習時の範囲の外の条件で上書きする
        c = self.condition
        if c in ("dark", "bright"):
            s = r.uniform(0.45, 0.55) if c == "dark" else r.uniform(1.5, 1.6)
            m.light_diffuse[ids["slit_light"]] = b["slit_diffuse"] * s
            m.vis.headlight.diffuse = b["head_diffuse"] * s
            m.vis.headlight.ambient = b["head_ambient"] * s
        elif c == "gray_iris":
            m.mat_rgba[ids["iris_mat"], :3] = r.uniform(0.42, 0.50)  # 灰色（学習時の5色に無い）
        elif c == "head_far":
            head_dx = r.choice([-1, 1]) * r.uniform(0.010, 0.014)
            head_dz = r.choice([-1, 1]) * r.uniform(0.008, 0.011)
            m.body_pos[ids["head"]] = b["head_pos"] + [head_dx, 0, head_dz]
        return {"head_dx_mm": head_dx * 1000, "head_dz_mm": head_dz * 1000, "pupil_r_mm": pupil_r * 1000}

    def reset(self, seed=None):
        """患者・照明をランダムに変え、細隙灯を瞳孔から少しずれた位置に置いて開始する。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.info = self._randomize()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 光軸が瞳孔中心を通る本体位置（＝正解の位置）
        aligned = self.error() + 0.0
        offset = np.array([self.rng.uniform(-0.015, 0.015), self.rng.uniform(-0.008, 0.008)])
        start = np.clip(aligned + offset, [-X_MAX, -Z_MAX], [X_MAX, Z_MAX])
        self.robot.grasp(*start)
        self.target = start.copy()
        self.steps = 0
        self.info.update({"start_offset_mm": (offset * 1000).tolist()})
        return self.observe()

    # ---------- 観測・ずれ ----------
    def observe(self):
        """細隙灯カメラの画像（img_size×img_size×3、uint8）。生徒AIが見るのはこれだけ。"""
        self.renderer.update_scene(self.data, camera="slit_cam")
        return self.renderer.render()

    def error(self):
        """瞳孔中心 − 光軸（細隙灯カメラの位置）の (x, z) [m]。シミュレータだけが知る特権情報。"""
        d = self.data
        p = d.site_xpos[self.ids["pupil_site"]]
        c = d.cam_xpos[self.ids["cam"]]
        return np.array([p[0] - c[0], p[2] - c[2]])

    def error_mm(self):
        return self.error() * 1000

    def close(self):
        """描画の後片付け（終了時のEGLの警告を防ぐ）。"""
        self.renderer.close()

    # ---------- 1ステップ ----------
    def step(self, action_mm, on_substep=None):
        """本体を (Δx, Δz)[mm] だけ動かす指令を出し、0.1秒進める。"""
        a = np.clip(np.asarray(action_mm, dtype=float), -MAX_STEP_MM, MAX_STEP_MM) / 1000
        prev = self.target.copy()
        self.target = np.clip(prev + a, [-X_MAX, -Z_MAX], [X_MAX, Z_MAX])
        for k in range(1, SUBSTEPS + 1):
            self.robot.command(*(prev + (self.target - prev) * k / SUBSTEPS))
            for _ in range(self.substep_n):
                mujoco.mj_step(self.model, self.data)
            if on_substep is not None:
                on_substep()
        self.steps += 1
        return self.observe()
