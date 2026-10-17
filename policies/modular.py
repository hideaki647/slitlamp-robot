"""モジュール型の方式（M4-A）：①画像処理で瞳孔を見つける → ②先生と同じ比例制御で動かす。学習は使わない。

M3 の生徒（画像からいきなり指令を出す End-to-End）と比べるための方式。
瞳孔そのものしか見ないので、顔の輪郭などの手がかりに頼らない。

使い方:
    policy = ModularPolicy()
    action = policy(obs)        # obs は細隙灯カメラの画像 (96, 96, 3) uint8
"""
import cv2
import numpy as np

from sim.env import MAX_STEP_MM

PX_PER_MM = 1.85          # 細隙灯カメラ 96×96・視野角15°・眼まで約0.197m での 1mm あたりの画素数
GAIN = 0.5                # 先生と同じ比例ゲイン
R_MIN_MM, R_MAX_MM = 1.2, 4.5   # 瞳孔の半径として認める範囲（ランダム化は 1.5〜4mm。少し余裕を持たせる）


_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _candidates(g, thr):
    """明るさ thr 以下の部分から、瞳孔らしい（大きさ・丸さが合う）かたまりを探す。"""
    mask = cv2.morphologyEx((g <= thr).astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE)  # 白い反射点を埋める
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    r_min, r_max = R_MIN_MM * PX_PER_MM, R_MAX_MM * PX_PER_MM
    out = []
    h, w = g.shape
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                 # 画面の端にかかるもの（背景・顔の影）は瞳孔ではない
        area = cv2.contourArea(c)
        if not (np.pi * r_min**2 <= area <= np.pi * r_max**2):
            continue
        # 丸さ：かたまりの面積 ÷ それを囲む最小の円の面積（円なら1）。小さいかたまりでも安定して測れる
        (_, _), r_enc = cv2.minEnclosingCircle(c)
        if area / (np.pi * r_enc**2) < 0.6:
            continue
        inside = np.zeros_like(mask)
        cv2.drawContours(inside, [c], -1, 1, -1)
        m = cv2.moments(c)
        out.append((g[inside > 0].mean(), m["m10"] / m["m00"], m["m01"] / m["m00"], np.sqrt(area / np.pi)))
    return out


def detect_pupil(img, step=3, max_shift_px=1.0, max_growth=1.6):
    """瞳孔の中心（画素の x, y）と半径（画素）を返す。見つからなければ None。

    しきい値を、画像のいちばん暗い値から少しずつ上げていき、同じかたまりを追いかける。
    瞳孔はまず暗い円として現れ、しきい値を上げると縁まで取れて中心が安定する。さらに上げると、
    虹彩の影の部分とつながって中心がずれたり、急に大きくなったりする。その直前の結果を使う。
    （1つのしきい値で決めると、明るい画像では虹彩ごと取り出す。M4-A で発見）
    """
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    lo, hi = int(g.min()), int(np.median(g))
    best = None
    for thr in range(lo + step, hi, step):
        cands = _candidates(g, thr)
        if best is None:
            if cands:
                best = min(cands)    # 最初に現れた瞳孔らしいかたまり（複数なら、いちばん暗いもの）
            continue
        # 前回のかたまりに最も近いものを追いかける
        near = [c for c in cands if np.hypot(c[1] - best[1], c[2] - best[2]) <= max_shift_px]
        if not near:
            break                    # 中心が大きく動いた・消えた（虹彩の影とつながった）→ 直前の結果を使う
        nxt = min(near, key=lambda c: np.hypot(c[1] - best[1], c[2] - best[2]))
        if nxt[3] ** 2 > max_growth * best[3] ** 2:
            break                    # 急に大きくなった（つながった）→ 直前の結果を使う
        best = nxt
    return None if best is None else best[1:]


def pupil_offset_mm(img):
    """瞳孔の中心 − 画面の中央 を (x, z)[mm] で返す（先生の「ずれ」と同じ向き）。見つからなければ None。
    画像の右 = +x、画像の上 = +z（画素の y は下向きなので符号を逆にする）。"""
    found = detect_pupil(img)
    if found is None:
        return None
    cx, cy, _ = found
    h, w = img.shape[:2]
    return np.array([(cx - (w - 1) / 2) / PX_PER_MM, -(cy - (h - 1) / 2) / PX_PER_MM])


# 校正：検出した中心は、いつも左下へ約0.1・0.2mm ずれる（細隙光が右上から当たり、瞳孔の左下が暗く写るため）。
# 実機で設置時に1回行う「校正」に相当する。右眼のお手本1,500枚で測った平均のずれ（scripts/m4a_detect_check.py --calib）。
CALIB_OFFSET_MM = np.array([-0.096, -0.206])


class ModularPolicy:
    """policy(obs, env=None) → 指令 (Δx, Δz)[mm]。env は見ない（画像だけで決める）。
    calibrate=False なら校正なし（検出したままの位置を使う）。"""

    def __init__(self, calibrate=True):
        self.offset = CALIB_OFFSET_MM if calibrate else np.zeros(2)
        self.misses = 0    # 瞳孔を見つけられなかった回数（評価用）
        self.calls = 0

    def __call__(self, obs, env=None):
        self.calls += 1
        e = pupil_offset_mm(obs)
        if e is None:                    # 見つからなければ動かない
            self.misses += 1
            return np.zeros(2)
        return np.clip(GAIN * (e - self.offset), -MAX_STEP_MM, MAX_STEP_MM)
