"""細隙灯アライメント環境：リセット（ランダム化）・1ステップ進める・撮影・ずれの計算。

1ステップ = 0.1秒。行動（action）は「本体をあとどれだけ動かすか」の差分 (Δx, Δz) [mm]。
M2 の先生、M3 の生徒（学習ポリシー）の両方がこの環境を使う。
"""
from pathlib import Path

import cv2
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


# 実機の不完全さ（M4-G）。既定はすべて「なし」＝これまでと同じ。
DISTURB_DEFAULT = {
    "delay": 0,        # 観測の遅れ [ステップ]（1ステップ=0.1秒）：方式が受け取る画像・ずれを何ステップ前のものにするか
    "act_noise": 0.0,  # 指令のばらつき [mm]：実行する指令に毎回足すランダムなずれの標準偏差
    "kp_scale": 1.0,   # モーターの個体差：アームの位置制御の強さ kp の倍率
    "img_noise": 0.0,  # 画像のノイズ：画素ごとのランダムなずれの標準偏差（0〜1、明るさ255に対する割合）
    "blur": 0.0,       # 画像のぼけ：ガウシアンぼかしの広がり [画素]
}


class SlitLampEnv:
    def __init__(self, img_size=96, seed=None, target_eye="R", condition="normal", disturb=None,
                 eye_mode=None, start="near", ipd_mm=None):
        """
        target_eye : 合わせる眼（"R"／"L"）。eye_mode="both" のときは患者ごとにランダム（M4-F）。
        start      : "near" = 目標の眼の近く ±15mm から（これまでと同じ）
                     "switch" = 反対の眼の近く ±5mm から（反対の眼に合わせた状態から移る、M4-F）
        ipd_mm     : 両眼の間の距離の範囲 (最小, 最大) [mm]。None なら 64mm 固定（これまでと同じ）
        """
        assert condition in CONDITIONS, condition
        assert start in ("near", "switch"), start
        self.condition = condition
        self.disturb = {**DISTURB_DEFAULT, **(disturb or {})}
        self.eye_mode, self.start_mode, self.ipd_mm = eye_mode, start, ipd_mm
        self.goal_eye = target_eye
        self.goal = 0 if target_eye == "R" else 1   # 生徒に渡す「指示」：0=右眼、1=左眼（M4-F）
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.robot = SlitLampRobot(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=img_size, width=img_size)
        self.rng = np.random.default_rng(seed)
        m = self.model

        self.ids = {
            "head": m.body("head").id,
            "pupil_site": m.site(f"pupil_{target_eye}_center").id,
            "pupil_sites": {e: m.site(f"pupil_{e}_center").id for e in "RL"},
            "eyes": {e: m.body(f"eye_{e}").id for e in "RL"},
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
            "eye_pos": {e: m.body_pos[self.ids["eyes"][e]].copy() for e in "RL"},
        }
        self.base_kp = m.actuator_gainprm[:, 0].copy()   # モーターの強さ（M4-G の個体差の基準）
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

        # 両眼切替（M4-F）：目標の眼・両眼の間の距離・開始位置。患者の乱数とは別の乱数を使い、
        # これらを使わないときは、これまでとまったく同じ患者・開始位置になるようにする
        srng = np.random.default_rng((seed if seed is not None else 0) + 3_000_000)
        m, ids = self.model, self.ids
        if self.ipd_mm is not None:
            half = srng.uniform(*self.ipd_mm) / 2000
            for e, sign in (("R", -1), ("L", 1)):
                m.body_pos[ids["eyes"][e]] = self.base["eye_pos"][e] * [0, 1, 1] + [sign * half, 0, 0]
            self.info["ipd_mm"] = half * 2000
        if self.eye_mode == "both":
            self.goal_eye = "R" if srng.random() < 0.5 else "L"
            self.goal = 0 if self.goal_eye == "R" else 1
        ids["pupil_site"] = ids["pupil_sites"][self.goal_eye]
        self.info.update({"goal_eye": self.goal_eye, "start": self.start_mode})

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 光軸が瞳孔中心を通る本体位置（＝正解の位置）
        aligned = self.error() + 0.0
        offset = np.array([self.rng.uniform(-0.015, 0.015), self.rng.uniform(-0.008, 0.008)])
        if self.start_mode == "switch":
            # 反対の眼に（ほぼ）合わせた状態から始める
            other = ids["pupil_sites"]["L" if self.goal_eye == "R" else "R"]
            p, c = self.data.site_xpos[other], self.data.cam_xpos[ids["cam"]]
            aligned = np.array([p[0] - c[0], p[2] - c[2]])
            offset = np.array([srng.uniform(-0.005, 0.005), srng.uniform(-0.003, 0.003)])
        start = np.clip(aligned + offset, [-X_MAX, -Z_MAX], [X_MAX, Z_MAX])
        self.robot.grasp(*start)
        self.target = start.copy()
        self.steps = 0
        self.info.update({"start_offset_mm": (offset * 1000).tolist()})

        # 実機の不完全さ（M4-G）。患者の乱数とは別の乱数を使い、乱れの有無で患者が変わらないようにする
        self.drng = np.random.default_rng((seed if seed is not None else 0) + 2_000_000)
        kp = self.base_kp * self.disturb["kp_scale"]
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp       # 位置制御モーター：力 = kp·(目標 − 角度) − kv·速度
        self._hist = [(self._degrade(self.observe()), self.error_mm())]
        return self._hist[0][0]

    def set_goal(self, eye):
        """診察の途中で目標の眼を変える（「次は左眼」の指示、M4-F の動画用）。"""
        self.goal_eye, self.goal = eye, (0 if eye == "R" else 1)
        self.ids["pupil_site"] = self.ids["pupil_sites"][eye]
        self._hist = [(img, self.error_mm()) for img, _ in self._hist]

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

    def observed_error_mm(self):
        """方式が受け取る（遅れた）ずれ [mm]。遅れが無ければ error_mm() と同じ。先生はこれを使う（M4-G）。"""
        return self._hist[0][1]

    def _degrade(self, img):
        """画像のぼけとノイズ（M4-G）。設定が無ければ何もしない。"""
        d = self.disturb
        if d["blur"] > 0:
            img = cv2.GaussianBlur(img, (0, 0), d["blur"])
        if d["img_noise"] > 0:
            img = np.clip(img + self.drng.normal(0, d["img_noise"] * 255, img.shape), 0, 255).astype(np.uint8)
        return img

    def close(self):
        """描画の後片付け（終了時のEGLの警告を防ぐ）。"""
        self.renderer.close()

    # ---------- 1ステップ ----------
    def step(self, action_mm, on_substep=None):
        """本体を (Δx, Δz)[mm] だけ動かす指令を出し、0.1秒進める。"""
        a = np.asarray(action_mm, dtype=float)
        if self.disturb["act_noise"] > 0:          # 指令のばらつき（M4-G）
            a = a + self.drng.normal(0, self.disturb["act_noise"], 2)
        a = np.clip(a, -MAX_STEP_MM, MAX_STEP_MM) / 1000
        prev = self.target.copy()
        self.target = np.clip(prev + a, [-X_MAX, -Z_MAX], [X_MAX, Z_MAX])
        for k in range(1, SUBSTEPS + 1):
            self.robot.command(*(prev + (self.target - prev) * k / SUBSTEPS))
            for _ in range(self.substep_n):
                mujoco.mj_step(self.model, self.data)
            if on_substep is not None:
                on_substep()
        self.steps += 1
        # 観測の遅れ（M4-G）：最新の画像とずれを積み、delay ステップ前のものを渡す
        self._hist.append((self._degrade(self.observe()), self.error_mm()))
        self._hist = self._hist[-(self.disturb["delay"] + 1):]
        return self._hist[0][0]
