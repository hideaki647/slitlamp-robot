"""瞼（まぶた）の形を作る（M4-E）。models/lids/ に OBJ ファイルを書き出し、scene.xml から読み込む。

瞼は、眼球（半径12mm）を包む半径 R の球面の一部（殻）として作る。眼球の中心を原点に、
x＝検者から見て右、y＝患者側（奥）、z＝上。正面（−y 側）から見た瞼の縁の形を、次の曲線で決める。

    上瞼の縁：z = h(c) · (1 − (x/W)²)   … 中央が高いアーチ（両端は内眼角・外眼角で z=0）
    下瞼の縁：z = −LOW · (1 − (x/W)²) + OVERLAP … 中央が低い（目尻・目頭で上瞼と少し重なり、瞼裂が閉じる）

上瞼は閉じ具合 c = 0〜1 ごとに N_LEVELS 個の形を作る（c=0：中央が瞳孔中心の 4.5mm 上＝普段、
c=1：下瞼の縁より下まで降りて眼を覆う）。シミュレータは c にいちばん近い形だけを表示する。

実行（slitlamp_robot フォルダで。形を変えたときだけ）:
    docker compose run --rm sim python scripts/make_lids.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "lids"

R = 0.0146          # 瞼の殻の半径 [m]（眼球 12mm、虹彩・瞳孔の前面 12.9mm より外。13.3mm では虹彩が透けた）
W = 0.013           # 瞼裂の半分の幅 [m]（内眼角・外眼角の x）
UP_OPEN = 0.0045    # 普段の上瞼の縁の中央の高さ [m]（瞳孔中心から 4.5mm 上＝虹彩の上端に 1.5mm かかる）
LOW = 0.0055        # 下瞼の縁の中央の低さ [m]（瞳孔中心から 5.5mm 下＝虹彩の下端に少し触れる）
OVERLAP = 0.0003    # 目頭・目尻で上下の瞼を重ねる量 [m]（隙間が見えないように）
CLOSED_EXTRA = 0.0008   # 閉じたとき、上瞼の縁が下瞼の縁より下に入る量 [m]
N_LEVELS = 13       # 上瞼の形の数（閉じ具合 0, 1/12, …, 1）
N_PHI, N_THETA = 64, 16
PHI_MAX = np.deg2rad(95)   # 左右に作る範囲（正面から ±95°。横の端は頭の中に隠れる）


def up_height(c):
    """閉じ具合 c の上瞼の縁の中央の高さ [m]。"""
    return UP_OPEN + c * (-LOW - CLOSED_EXTRA - UP_OPEN)


def edge_theta(phi, z_of_x):
    """方位 phi（正面から右回り）で、縁の曲線 z = z_of_x(x) と球面が交わる仰角 theta を求める。"""
    theta = 0.0
    for _ in range(30):      # x が theta によって変わるので、くり返して求める
        x = R * np.cos(theta) * np.sin(phi)
        theta = np.arcsin(np.clip(z_of_x(x) / R, -1, 1))
    return theta


def shell(theta_from, theta_to):
    """方位ごとに、仰角 theta_from(phi)〜theta_to(phi) の球面の帯を三角形の網にする（表と裏の両方の向き）。"""
    phis = np.linspace(-PHI_MAX, PHI_MAX, N_PHI)
    verts, faces = [], []
    for phi in phis:
        for t in np.linspace(theta_from(phi), theta_to(phi), N_THETA):
            verts.append((R * np.cos(t) * np.sin(phi), -R * np.cos(t) * np.cos(phi), R * np.sin(t)))
    for i in range(N_PHI - 1):
        for j in range(N_THETA - 1):
            a, b = i * N_THETA + j, i * N_THETA + j + 1
            c, d = a + N_THETA, b + N_THETA
            for tri in ((a, c, d), (a, d, b)):
                faces += [tri, tri[::-1]]
    return verts, faces


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def arch(height, offset=0.0):
    """中央の高さ height、両端（|x|=W）で offset のアーチ。|x|>W では offset（瞼裂の外）。"""
    return lambda x: offset + height * max(0.0, 1 - (x / W) ** 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    top = lambda phi: np.pi / 2
    bottom = lambda phi: -np.pi / 2
    for k in range(N_LEVELS):
        c = k / (N_LEVELS - 1)
        z_up = arch(up_height(c))
        write_obj(OUT / f"lid_up_{k:02d}.obj", *shell(lambda phi: edge_theta(phi, z_up), top))
    z_low = arch(-LOW, OVERLAP)
    write_obj(OUT / "lid_low.obj", *shell(bottom, lambda phi: edge_theta(phi, z_low)))
    print("wrote", N_LEVELS, "upper lids + lower lid to", OUT.relative_to(ROOT))
    print("upper lid center height [mm] by level:", [round(up_height(k / (N_LEVELS - 1)) * 1000, 1)
                                                     for k in range(N_LEVELS)])


if __name__ == "__main__":
    main()
