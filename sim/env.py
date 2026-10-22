"""細隙灯アライメント環境：リセット（ランダム化）・1ステップ進める・撮影・ずれの計算。

1ステップ = 0.1秒。行動（action）は「本体をあとどれだけ動かすか」の差分 (Δx, Δz) [mm]。
M2 の先生、M3 の生徒（学習ポリシー）の両方がこの環境を使う。
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sim.slitlamp_api import X_MAX, Y_MAX, Z_MAX, SlitLampRobot

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


# 瞬き（M4-E）
LID_LEVELS = 13            # 上瞼の形の数（scripts/make_lids.py と同じ）
LID_HIDE = np.array([0.0, 0.05, 0.0])   # 使わない瞼を眼球の奥へ隠すずらし量 [m]（scene.xml と同じ）
BLINK_CLOSE, BLINK_OPEN = 0.10, 0.15   # 閉じる時間・開く時間 [s]
PUPIL_HIDDEN_AT = 0.42     # この閉じ具合を超えると、上瞼の縁の中央が瞳孔中心より下（瞳孔が半分以上隠れる）
EVAL_BLINKS = (1.0, 2.5)   # 評価用：この時刻に必ず瞬きする [s]（閉じたままの時間 0.1秒）

# ピント合わせ（M4-H）
FOCUS_TOL_MM = 0.5           # 前後のずれがこれ未満なら「ピントが合っている」
BLUR_PX_PER_MM = 1.0         # 焦点からのずれ 1mm あたりのぼかしの広がり（ガウシアンの σ）[画素]
BLUR_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # 作っておくぼかしの強さ [画素]。画素ごとに近いものを選ぶ

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
                 eye_mode=None, start="near", ipd_mm=None, blink=None, focus=None):
        """
        target_eye : 合わせる眼（"R"／"L"）。eye_mode="both" のときは患者ごとにランダム（M4-F）。
        start      : "near" = 目標の眼の近く ±15mm から（これまでと同じ）
                     "switch" = 反対の眼の近く ±5mm から（反対の眼に合わせた状態から移る、M4-F）
        ipd_mm     : 両眼の間の距離の範囲 (最小, 最大) [mm]。None なら 64mm 固定（これまでと同じ）
        blink      : 右眼の瞼と瞬き（M4-E）。None = 瞼なし（これまでと同じ）／"rest" = 瞼はあるが瞬きしない／
                     "random" = ランダムな時刻に瞬き（学習用）／"eval" = 1.0秒と2.5秒に必ず瞬き（評価用）
        focus      : ピント合わせ（M4-H）。None = ボケなし・前後は動かさない（これまでと同じ）／
                     "back" = 焦点より 3〜8mm 手前（検者側）から始める（臨床でよく行う手順）／
                     "front" = 焦点より 3〜8mm 奥（近づきすぎ）から始める。指令は (Δx, Δz, Δy) の3つになる
        """
        assert blink in (None, "rest", "random", "eval"), blink
        assert focus in (None, "back", "front"), focus
        self.blink, self.focus = blink, focus
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
        # 瞼（M4-E）：上瞼13個＋下瞼。scene.xml では眼球の奥に隠してあり、表示するものだけ眼の位置へ動かす
        self.lid_ids = [m.geom(f"lid_up_{k:02d}").id for k in range(LID_LEVELS)] + [m.geom("lid_low").id]
        self.lid_hidden = m.geom_pos[self.lid_ids].copy()
        self.lid_closure = 0.0
        self.blinks = []           # 瞬きの予定 [(開始, 閉じたままの時間)]
        self.substep_n = round(CONTROL_DT / SUBSTEPS / m.opt.timestep)  # 1分割あたりの物理ステップ数
        # ピント合わせ（M4-H）：焦点までの距離＝標準の姿勢で、細隙灯カメラから瞳孔中心までの前後の距離
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        self.focus_dist = d0.site_xpos[self.ids["pupil_site"]][1] - d0.cam_xpos[self.ids["cam"]][1]
        self.target_y = 0.0
        if focus:
            self.depth_renderer = mujoco.Renderer(m, height=img_size, width=img_size)
            self.depth_renderer.enable_depth_rendering()
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
        start_y = 0.0
        if self.focus:
            # 患者の前後の座り方 ±5mm、開始は焦点より 3〜8mm 手前（back）または奥（front）。患者の乱数とは別の乱数
            frng = np.random.default_rng((seed if seed is not None else 0) + 5_000_000)
            m.body_pos[ids["head"]][1] = self.base["head_pos"][1] + frng.uniform(-0.005, 0.005)
            mujoco.mj_forward(self.model, self.data)
            p, c = self.data.site_xpos[ids["pupil_site"]], self.data.cam_xpos[ids["cam"]]
            aligned_y = (p[1] - c[1]) - self.focus_dist          # ピントが合う本体の前後位置
            gap = frng.uniform(0.003, 0.008)
            start_y = float(np.clip(aligned_y + (-gap if self.focus == "back" else gap), -Y_MAX, Y_MAX))
            self.info["start_focus_mm"] = (start_y - aligned_y) * 1000
        self.robot.grasp(*start, y=start_y)
        self.target = start.copy()
        self.target_y = start_y
        self.steps = 0
        self.info.update({"start_offset_mm": (offset * 1000).tolist()})

        # 実機の不完全さ（M4-G）。患者の乱数とは別の乱数を使い、乱れの有無で患者が変わらないようにする
        self.drng = np.random.default_rng((seed if seed is not None else 0) + 2_000_000)
        kp = self.base_kp * self.disturb["kp_scale"]
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp       # 位置制御モーター：力 = kp·(目標 − 角度) − kv·速度
        brng = np.random.default_rng((seed if seed is not None else 0) + 4_000_000)
        if self.blink == "random":           # 学習用：最初は0.3〜1.5秒後、その後は1.5〜4秒おき
            t, self.blinks = brng.uniform(0.3, 1.5), []
            while t < 8.0:
                self.blinks.append((t, brng.uniform(0.0, 0.15)))
                t += brng.uniform(1.5, 4.0)
        else:
            self.blinks = [(t, 0.1) for t in EVAL_BLINKS] if self.blink == "eval" else []
        self.info["blinks"] = [round(b[0], 2) for b in self.blinks]
        self._set_lid(0.0)
        self._hist = [(self._degrade(self.observe()), self.error_mm())]
        return self._hist[0][0]

    # ---------- 瞼・瞬き（M4-E） ----------
    def closure_at(self, t):
        """時刻 t [s] の上瞼の閉じ具合（0=普段、1=完全に閉じる）。"""
        c = 0.0
        for start, hold in self.blinks:
            dt = t - start
            if 0 <= dt < BLINK_CLOSE:
                c = max(c, dt / BLINK_CLOSE)
            elif BLINK_CLOSE <= dt < BLINK_CLOSE + hold:
                c = 1.0
            elif BLINK_CLOSE + hold <= dt < BLINK_CLOSE + hold + BLINK_OPEN:
                c = max(c, 1 - (dt - BLINK_CLOSE - hold) / BLINK_OPEN)
        return c

    def _set_lid(self, t):
        """時刻 t の瞼を表示する（いちばん近い形の上瞼と、下瞼）。瞼なしの設定なら全部隠す。"""
        m = self.model
        m.geom_pos[self.lid_ids] = self.lid_hidden
        self.lid_closure = self.closure_at(t) if self.blink else 0.0
        if self.blink:
            k = int(round(self.lid_closure * (LID_LEVELS - 1)))
            for g in (self.lid_ids[k], self.lid_ids[-1]):
                m.geom_pos[g] = self.lid_hidden[self.lid_ids.index(g)] - LID_HIDE
        mujoco.mj_kinematics(m, self.data)

    @property
    def pupil_hidden(self):
        """瞳孔が瞼で半分以上隠れているか（特権情報。先生が「待つ」かどうかの判断に使う）。"""
        return self.lid_closure > PUPIL_HIDDEN_AT

    def set_goal(self, eye):
        """診察の途中で目標の眼を変える（「次は左眼」の指示、M4-F の動画用）。"""
        self.goal_eye, self.goal = eye, (0 if eye == "R" else 1)
        self.ids["pupil_site"] = self.ids["pupil_sites"][eye]
        self._hist = [(img, self.error_mm()) for img, _ in self._hist]

    # ---------- 観測・ずれ ----------
    def observe(self):
        """細隙灯カメラの画像（img_size×img_size×3、uint8）。生徒AIが見るのはこれだけ。
        ピント合わせ（M4-H）のときは、焦点からのずれに応じて画素ごとにぼかす。"""
        self.renderer.update_scene(self.data, camera="slit_cam")
        img = self.renderer.render()
        if self.focus:
            img = self._defocus(img)
        return img

    def _defocus(self, img):
        """深度画像（カメラから各画素までの前後の距離）を使い、焦点からのずれが大きい画素ほど強くぼかす。
        ボケの広がり σ = BLUR_PX_PER_MM × |距離 − 焦点距離|[mm]。強さの違うぼかしを数段階作り、画素ごとに選ぶ。"""
        self.depth_renderer.update_scene(self.data, camera="slit_cam")
        depth = self.depth_renderer.render()
        sigma = np.clip(BLUR_PX_PER_MM * np.abs(depth - self.focus_dist) * 1000, 0, BLUR_LEVELS[-1])
        level = np.abs(sigma[..., None] - np.array(BLUR_LEVELS)).argmin(-1)
        out = img.copy()
        for k, s in enumerate(BLUR_LEVELS):
            if s > 0 and (level == k).any():
                out[level == k] = cv2.GaussianBlur(img, (0, 0), s)[level == k]
        return out

    def error(self):
        """瞳孔中心 − 光軸（細隙灯カメラの位置）の (x, z) [m]。シミュレータだけが知る特権情報。"""
        d = self.data
        p = d.site_xpos[self.ids["pupil_site"]]
        c = d.cam_xpos[self.ids["cam"]]
        return np.array([p[0] - c[0], p[2] - c[2]])

    def error_mm(self):
        """ずれ [mm]。ピント合わせ（M4-H）のときは (x, z, y) の3つ。y = 瞳孔までの前後の距離 − 焦点距離
        （正なら焦点より手前＝もっと奥へ押す必要がある）。指令 (Δx, Δz, Δy) と同じ並び。"""
        e = self.error() * 1000
        if self.focus:
            d = self.data
            gap = d.site_xpos[self.ids["pupil_site"]][1] - d.cam_xpos[self.ids["cam"]][1] - self.focus_dist
            e = np.append(e, gap * 1000)
        return e

    def is_aligned(self, err_mm=None):
        """合っているか：左右・上下が SUCCESS_MM 未満（ピント合わせのときは、さらに前後が FOCUS_TOL_MM 未満）。"""
        e = self.error_mm() if err_mm is None else err_mm
        ok = np.linalg.norm(e[:2]) < SUCCESS_MM
        return bool(ok and (len(e) < 3 or abs(e[2]) < FOCUS_TOL_MM))

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
        if self.focus:
            self.depth_renderer.close()

    # ---------- 1ステップ ----------
    def step(self, action_mm, on_substep=None):
        """本体を (Δx, Δz)[mm] だけ動かす指令を出し、0.1秒進める。"""
        a = np.asarray(action_mm, dtype=float)
        if self.disturb["act_noise"] > 0:          # 指令のばらつき（M4-G）
            a = a + self.drng.normal(0, self.disturb["act_noise"], 2)
        a = np.clip(a, -MAX_STEP_MM, MAX_STEP_MM) / 1000
        prev, prev_y = self.target.copy(), self.target_y
        self.target = np.clip(prev + a[:2], [-X_MAX, -Z_MAX], [X_MAX, Z_MAX])
        if self.focus and len(a) > 2:          # 前後（ピント合わせ、M4-H）
            self.target_y = float(np.clip(prev_y + a[2], -Y_MAX, Y_MAX))
        for k in range(1, SUBSTEPS + 1):
            f = k / SUBSTEPS
            self.robot.command(*(prev + (self.target - prev) * f), y=prev_y + (self.target_y - prev_y) * f)
            for _ in range(self.substep_n):
                mujoco.mj_step(self.model, self.data)
            if self.blink:                     # 瞼は 0.02秒ごとに動かす（半分閉じた画像も生じる）
                self._set_lid((self.steps + k / SUBSTEPS) * CONTROL_DT)
            if on_substep is not None:
                on_substep()
        self.steps += 1
        # 観測の遅れ（M4-G）：最新の画像とずれを積み、delay ステップ前のものを渡す
        self._hist.append((self._degrade(self.observe()), self.error_mm()))
        self._hist = self._hist[-(self.disturb["delay"] + 1):]
        return self._hist[0][0]
