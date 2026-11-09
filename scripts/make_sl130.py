"""Zeiss SL130 をモデルにした細隙灯の見た目を作る（M6-①）。

scene.xml を元に models/scene_sl130.xml を組み立てる（scene.xml 自体は変えない）。
- 今までの細隙灯の部品は、描画グループ3（表示しない）に移すだけで残す → 重さ・動きの計算はまったく同じ。
- 新しい部品は「重さ0・衝突なし」の見た目だけの部品（default class "sl"）。
- 倍率ドラム（左右両側、一緒に回る）・照明のつまみ類・明るさのつまみは、回転の関節だけ付ける（働きは後の段階）。
- 細隙灯カメラ（slit_cam）と細隙光（slit_light）の位置は変えない。新しい部品はすべてカメラより後ろ（y < 0.125）にある。
- 新しい部品は描画グループ2。sim/env.py は細隙灯カメラの画像を撮るときだけ、グループ2を隠してグループ3を表示する
  （部品の影が顔に落ちるため）→ 細隙灯カメラの画像もこれまでと同じ。

倍率ドラムの目盛り（ユーザー提供の写真より）：検者側（-y）の白い点の前に来た数字が今の倍率。
ドラムの面の上で、患者側（+y）を0°として上回り（+x から見て反時計回り）に
  0°:12, 60°:32, 120°:20, 180°:12（白い点の位置＝基本）, 240°:5, 300°:8
ドラムを +60° 回すごとに 12 → 20 → 32 → 12 → 8 → 5 → 12 と変わる。左右の面とも同じ角度に同じ数字。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/make_sl130.py
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = MODELS / "sl130"

DRUM_R = 0.026        # 倍率ドラムの半径 [m]
DRUM_H = 0.003        # ドラムの厚みの半分 [m]
DRUM_X = 0.0352       # ドラムの中心の左右の位置（顕微鏡の本体の側面 0.032 の少し外）
MARKS = {0: "12", 60: "32", 120: "20", 180: "12", 240: "5", 300: "8"}
TEX = 512
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def dial_texture(side):
    """ドラムの面の画像。side=+1：右側面（+x から見る）、-1：左側面（-x から見る）。
    面の上の座標 (u, v)：見る人から見て右が u、上が v。右側面は u=+y、左側面は u=-y。"""
    img = Image.new("RGB", (TEX, TEX), (20, 20, 20))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 58)
    for ang, text in MARKS.items():
        y, z = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        u, v = side * y * 0.76, z * 0.76
        # 数字の上が中心を向くように回す（実機の写真どおり。下側の 5・8 が正立して読める）
        tile = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((60, 45), text, font=font, fill=(235, 235, 235, 255), anchor="mm")
        tile = tile.rotate(np.degrees(np.arctan2(v, u)) + 90, expand=True, resample=Image.BICUBIC)
        cx, cy = (u * 0.5 + 0.5) * TEX, (0.5 - v * 0.5) * TEX
        img.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)
    return img


def dial_mesh(side, n=64):
    """厚みのある円盤（軸は x）。外側の面に目盛りの画像を貼る。uv は面の座標 (u, v) から。"""
    v_list, vt_list, faces = [], [], []
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xo, xi = side * DRUM_H, -side * DRUM_H                 # 外側の面・内側の面の x
    def add(x, y, z):
        v_list.append((x, y, z))
        vt_list.append((side * y / (2 * DRUM_R) + 0.5, z / (2 * DRUM_R) + 0.5))
        return len(v_list)
    co, ci = add(xo, 0, 0), add(xi, 0, 0)
    ro = [add(xo, DRUM_R * np.cos(a), DRUM_R * np.sin(a)) for a in ang]
    ri = [add(xi, DRUM_R * np.cos(a), DRUM_R * np.sin(a)) for a in ang]
    for k in range(n):
        a, b = k, (k + 1) % n
        # 面の向き：外側から見て反時計回り（右側面では y→z の向き、左側面では逆）
        if side > 0:
            faces += [(co, ro[a], ro[b]), (ci, ri[b], ri[a]), (ro[a], ri[a], ri[b]), (ro[a], ri[b], ro[b])]
        else:
            faces += [(co, ro[b], ro[a]), (ci, ri[a], ri[b]), (ro[a], ri[b], ri[a]), (ro[a], ro[b], ri[b])]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v_list]
    lines += [f"vt {u:.5f} {w:.5f}" for u, w in vt_list]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


ASSETS = """    <!-- ===== Zeiss SL130 の見た目（M6-①、scripts/make_sl130.py が作る） ===== -->
    <material name="sl_white" rgba="0.86 0.87 0.86 1"/>
    <material name="sl_black" rgba="0.07 0.07 0.07 1"/>
    <material name="sl_knurl" rgba="0.16 0.16 0.16 1"/>
    <material name="sl_glass" rgba="0.18 0.22 0.27 1" specular="0.8" shininess="0.9"/>
    <material name="sl_steel" rgba="0.72 0.73 0.75 1" specular="0.6" shininess="0.6"/>
    <material name="sl_red"   rgba="0.85 0.12 0.10 1"/>
    <texture name="dial_R_tex" type="2d" file="sl130/dial_R.png"/>
    <texture name="dial_L_tex" type="2d" file="sl130/dial_L.png"/>
    <material name="dial_R_mat" texture="dial_R_tex"/>
    <material name="dial_L_mat" texture="dial_L_tex"/>
    <mesh name="dial_R" file="sl130/dial_R.obj"/>
    <mesh name="dial_L" file="sl130/dial_L.obj"/>
"""

DEFAULTS = """    <!-- SL130 の見た目だけの部品：重さ0・衝突なし。つまみの関節は軽く、少し粘りを持たせる。
         描画グループ2：細隙灯カメラの画像を撮るときだけ隠す（室内灯の影が顔に落ちて画像が変わらないように、sim/env.py） -->
    <default class="sl">
      <geom contype="0" conaffinity="0" density="0" material="sl_white" group="2"/>
      <joint type="hinge" damping="0.001" armature="0.0001"/>
    </default>
"""

# 土台（台に固定）：T字形の白い板と、左右のレール
WORLD_BASE = """
    <!-- SL130 の土台（台に固定、見た目のみ）：T字形の板、奥の左右のレール -->
    <body name="sl_baseplate" pos="0 0 0.75" childclass="sl">
      <geom name="sl_plate_main"  type="box" pos="0 0.07 0.003"  size="0.20 0.24 0.003"/>
      <geom name="sl_plate_front" type="box" pos="0 -0.215 0.003" size="0.075 0.045 0.003"/>
      <geom name="sl_rail"        type="cylinder" fromto="-0.19 -0.14 0.012 0.19 -0.14 0.012" size="0.005" material="sl_steel"/>
      <geom name="sl_rail_end_L"  type="box" pos="-0.19 -0.14 0.012" size="0.012 0.02 0.012"/>
      <geom name="sl_rail_end_R"  type="box" pos=" 0.19 -0.14 0.012" size="0.012 0.02 0.012"/>
    </body>
"""

# 基台（左右・前後に動く）：本体の台と、操作棒のまわりの丸い台座、明るさのつまみ
CARRIAGE = """
      <!-- SL130 の基台（見た目のみ）：本体の台、操作棒の丸い台座、明るさのつまみ -->
      <geom name="sl_carriage"  class="sl" type="box" pos="0 0.0 0.022" size="0.05 0.07 0.022"/>
      <geom name="sl_front"     class="sl" type="box" pos="0 -0.06 0.018" size="0.045 0.03 0.018"/>
      <geom name="sl_js_socket" class="sl" type="cylinder" pos="0 -0.09 0.018" size="0.036 0.018"/>
      <geom name="sl_js_hole"   class="sl" type="cylinder" pos="0 -0.09 0.037" size="0.016 0.001" material="sl_black"/>
      <body name="sl_brightness" pos="0.042 -0.075 0.036" childclass="sl">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
        <joint name="brightness" axis="0 0 1"/>
        <geom name="sl_brightness_knob" type="cylinder" pos="0 0 0.009" size="0.008 0.009" material="sl_black"/>
      </body>
"""

# 操作棒：太い黒い棒（つかむ点 js_grip の位置は今と同じ）
HANDLE = """
        <!-- SL130 の操作棒（見た目のみ）：太い黒い棒と、ローレットの付いた頭 -->
        <geom name="sl_js_handle" class="sl" type="cylinder" fromto="0 0 0.0 0 0 0.10" size="0.012" material="sl_black"/>
        <geom name="sl_js_head"   class="sl" type="cylinder" fromto="0 0 0.09 0 0 0.125" size="0.0155" material="sl_knurl"/>
"""

# 支柱の位置：共通の回転軸、顕微鏡の「コ」の字の腕、照明の本体と筒（つまみは関節つき）
COLUMN = """
        <!-- SL130 の共通の回転軸と、顕微鏡の「コ」の字の腕（見た目のみ） -->
        <geom name="sl_pivot"     class="sl" type="cylinder" pos="0 0.03 0.025" size="0.032 0.025"/>
        <geom name="sl_pivot_scale" class="sl" type="cylinder" pos="0 0.03 0.038" size="0.0325 0.004" material="sl_steel"/>
        <geom name="sl_arm_low"   class="sl" type="box" pos="0 -0.005 0.03"  size="0.017 0.04 0.014"/>
        <geom name="sl_arm_up"    class="sl" type="box" pos="0 -0.033 0.125" size="0.017 0.015 0.095"/>
        <geom name="sl_arm_top"   class="sl" type="box" pos="0 -0.012 0.212" size="0.017 0.032 0.013"/>
        <!-- 照明の本体：顕微鏡の腕の前（患者側）。両側に黒いつまみ、前の下に黒い格子 -->
        <geom name="sl_illum_box" class="sl" type="box" pos="0 0.085 0.105" size="0.028 0.032 0.05"/>
        <geom name="sl_illum_grille" class="sl" type="box" pos="0 0.1175 0.078" size="0.02 0.0015 0.02" material="sl_black"/>
        <geom name="sl_illum_tube" class="sl" type="cylinder" fromto="0 0.09 0.155 0 0.09 0.236" size="0.0125" material="sl_black"/>
        <!-- 照明の筒のローレット：回すと光の当たる位置が上下に動く -->
        <body name="sl_illum_vert" pos="0 0.09 0.2" childclass="sl">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
          <joint name="illum_vert" axis="0 0 1"/>
          <geom name="sl_illum_knurl" type="cylinder" size="0.0145 0.016" material="sl_knurl"/>
          <geom name="sl_illum_knurl_mark" type="box" pos="0 -0.0145 0" size="0.0015 0.001 0.016" material="sl_white"/>
        </body>
        <geom name="sl_illum_head" class="sl" type="box" pos="0 0.09 0.243" size="0.011 0.011 0.007" material="sl_black"/>
        <!-- 照明のつまみ（左右両側、一緒に回る）：上から 光の左右（偏心）、（未使用）、スリットの幅、（未使用） -->
        <body name="sl_illum_decenter" pos="0 0.075 0.142" childclass="sl">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
          <joint name="illum_decenter" axis="1 0 0"/>
          <geom name="sl_knob_dec_R" type="cylinder" fromto=" 0.028 0 0  0.04 0 0" size="0.007" material="sl_black"/>
          <geom name="sl_knob_dec_L" type="cylinder" fromto="-0.028 0 0 -0.04 0 0" size="0.007" material="sl_black"/>
        </body>
        <geom name="sl_knob_2_R" class="sl" type="cylinder" fromto=" 0.028 0.075 0.12  0.039 0.075 0.12" size="0.008" material="sl_black"/>
        <geom name="sl_knob_2_L" class="sl" type="cylinder" fromto="-0.028 0.075 0.12 -0.039 0.075 0.12" size="0.008" material="sl_black"/>
        <body name="sl_slit_width" pos="0 0.075 0.093" childclass="sl">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
          <joint name="slit_width" axis="1 0 0"/>
          <geom name="sl_knob_w_R" type="cylinder" fromto=" 0.028 0 0  0.043 0 0" size="0.0105" material="sl_black"/>
          <geom name="sl_knob_w_L" type="cylinder" fromto="-0.028 0 0 -0.043 0 0" size="0.0105" material="sl_black"/>
        </body>
        <geom name="sl_knob_4_R" class="sl" type="cylinder" fromto=" 0.028 0.075 0.067  0.039 0.075 0.067" size="0.0075" material="sl_black"/>
        <geom name="sl_knob_4_L" class="sl" type="cylinder" fromto="-0.028 0.075 0.067 -0.039 0.075 0.067" size="0.0075" material="sl_black"/>
"""

# 顕微鏡：本体、患者側の黒い面と対物レンズ、接眼部、倍率ドラム（左右両側、関節つき）
MICROSCOPE = """
          <!-- SL130 の顕微鏡（見た目のみ）：本体、患者側の黒い面と対物レンズ、接眼部 -->
          <geom name="sl_mic_body"  class="sl" type="box" pos="0 0.01 0" size="0.032 0.048 0.034"/>
          <geom name="sl_mic_front" class="sl" type="box" pos="0 0.062 0" size="0.032 0.004 0.034" material="sl_black"/>
          <geom name="sl_objective" class="sl" type="cylinder" fromto="0 0.066 0 0 0.072 0" size="0.021" material="sl_glass"/>
          <geom name="sl_bino_box"  class="sl" type="box" pos="0 -0.05 0.014" size="0.026 0.014 0.03"/>
          <geom name="sl_bino_L"    class="sl" type="cylinder" fromto="-0.017 -0.055 0.03 -0.02 -0.108 0.064" size="0.0115"/>
          <geom name="sl_bino_R"    class="sl" type="cylinder" fromto=" 0.017 -0.055 0.03  0.02 -0.108 0.064" size="0.0115"/>
          <geom name="sl_ocular_L"  class="sl" type="cylinder" fromto="-0.02 -0.108 0.064 -0.021 -0.13 0.078" size="0.0135" material="sl_black"/>
          <geom name="sl_ocular_R"  class="sl" type="cylinder" fromto=" 0.02 -0.108 0.064  0.021 -0.13 0.078" size="0.0135" material="sl_black"/>
          <!-- 倍率ドラムの目印：検者側の白い点（左右両側）。この前に来た数字が今の倍率 -->
          <geom name="sl_drum_index_R" class="sl" type="sphere" pos=" 0.0322 -0.0195 0" size="0.0012" material="sl_steel"/>
          <geom name="sl_drum_index_L" class="sl" type="sphere" pos="-0.0322 -0.0195 0" size="0.0012" material="sl_steel"/>
          <!-- 倍率ドラム（左右両側が1つの部品＝一緒に回る）。+60°ごとに 12→20→32→12→8→5→12 -->
          <body name="sl_mag_drum" pos="0 0.01 0" childclass="sl">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
            <joint name="mag_drum" axis="1 0 0"/>
            <geom name="sl_drum_R" type="mesh" mesh="dial_R" pos=" 0.0352 0 0" material="dial_R_mat"/>
            <geom name="sl_drum_L" type="mesh" mesh="dial_L" pos="-0.0352 0 0" material="dial_L_mat"/>
            <geom name="sl_drum_bar_R" type="box" pos=" 0.041 0 0" size="0.003 0.0135 0.004" material="sl_black"/>
            <geom name="sl_drum_bar_L" type="box" pos="-0.041 0 0" size="0.003 0.0135 0.004" material="sl_black"/>
            <geom name="sl_drum_nub_R" type="sphere" pos=" 0.0442 0 0" size="0.0022" material="sl_knurl"/>
            <geom name="sl_drum_nub_L" type="sphere" pos="-0.0442 0 0" size="0.0022" material="sl_knurl"/>
          </body>
"""

# 顎台・額当て：左右の柱の台座と赤い印、上の横棒と弓なりの額当て、顎台の横棒と黒い顎受け、右の柱の黒い筒
HEADREST = """
      <!-- SL130 の顎台・額当て（見た目のみ）。赤い印は眼の高さ（外眼角の目安） -->
      <geom name="sl_pole_foot_L" class="sl" type="box" pos="-0.13 0 0.012" size="0.022 0.03 0.012"/>
      <geom name="sl_pole_foot_R" class="sl" type="box" pos=" 0.13 0 0.012" size="0.022 0.03 0.012"/>
      <geom name="sl_pole_L" class="sl" type="cylinder" fromto="-0.13 0 0 -0.13 0 0.45" size="0.0085" material="sl_steel"/>
      <geom name="sl_pole_R" class="sl" type="cylinder" fromto=" 0.13 0 0  0.13 0 0.45" size="0.0085" material="sl_steel"/>
      <geom name="sl_mark_L" class="sl" type="cylinder" pos="-0.13 0 0.30" size="0.0088 0.0015" material="sl_red"/>
      <geom name="sl_mark_R" class="sl" type="cylinder" pos=" 0.13 0 0.30" size="0.0088 0.0015" material="sl_red"/>
      <geom name="sl_top_cap_L" class="sl" type="box" pos="-0.13 0 0.455" size="0.018 0.02 0.01"/>
      <geom name="sl_top_cap_R" class="sl" type="box" pos=" 0.13 0 0.455" size="0.018 0.02 0.01"/>
      <geom name="sl_band_C"  class="sl" type="capsule" fromto="-0.045 -0.042 0.44  0.045 -0.042 0.44" size="0.009"/>
      <geom name="sl_band_L"  class="sl" type="capsule" fromto="-0.13 0 0.455 -0.045 -0.042 0.44" size="0.009"/>
      <geom name="sl_band_R"  class="sl" type="capsule" fromto=" 0.13 0 0.455  0.045 -0.042 0.44" size="0.009"/>
      <geom name="sl_chin_bar" class="sl" type="box" pos="0 0 0.185" size="0.14 0.012 0.008"/>
      <geom name="sl_chin_cup" class="sl" type="ellipsoid" pos="0 -0.01 0.2" size="0.038 0.03 0.012" material="sl_black"/>
      <geom name="sl_chin_knob" class="sl" type="cylinder" fromto="0.13 0 0.03 0.13 0 0.16" size="0.013" material="sl_black"/>
"""

# 今までの細隙灯・顎台の部品：表示しない（描画グループ3）が、重さはそのまま残す
HIDE = ["base_plate", "js_shaft", "js_knob", "column", "scope_tube", "eyepiece_L", "eyepiece_R",
        "illum_tube", "chin_cup", "forehead", "cr_pole_L", "cr_pole_R"]


def insert_after(xml, anchor_name, block):
    """name="anchor_name" の行の直後に block を入れる。"""
    m = re.search(rf'^.*name="{anchor_name}".*$', xml, flags=re.M)
    assert m, anchor_name
    return xml[:m.end()] + block.rstrip("\n") + xml[m.end():]


def build_scene():
    xml = (MODELS / "scene.xml").read_text(encoding="utf-8")
    xml = xml.replace("<mujoco model=\"slitlamp_scene\">",
                      "<!-- 自動生成：scripts/make_sl130.py（直接編集しない）。scene.xml に SL130 の見た目を足したもの -->\n"
                      "<mujoco model=\"slitlamp_scene_sl130\">", 1)
    # 影の範囲などは、シーン全体の大きさ（statistic）から自動で決まる。部品を足すと変わり、影の縁がずれて
    # 細隙灯カメラの画像が変わるので、元の scene.xml と同じ値に固定する
    import mujoco
    st = mujoco.MjModel.from_xml_path(str(MODELS / "scene.xml")).stat
    center = " ".join(f"{c:.9g}" for c in st.center)
    stat = (f'  <!-- scene.xml と同じ値に固定（影の範囲を同じにする） -->\n'
            f'  <statistic center="{center}" extent="{st.extent:.9g}" meansize="{st.meansize:.9g}"'
            f' meanmass="{st.meanmass:.9g}" meaninertia="{st.meaninertia:.9g}"/>')
    xml = xml.replace('  <compiler angle="degree"/>', '  <compiler angle="degree"/>\n' + stat, 1)
    for name in HIDE:
        xml, n = re.subn(rf'(<geom name="{name}" )', r'\1group="3" ', xml)
        assert n == 1, name
    xml = xml.replace("  </default>", DEFAULTS + "  </default>", 1)
    xml = xml.replace("  </asset>", ASSETS + "  </asset>", 1)
    xml = xml.replace("    <!-- ===== 細隙灯 ===== -->", WORLD_BASE.strip("\n") + "\n\n    <!-- ===== 細隙灯 ===== -->", 1)
    xml = insert_after(xml, "base_plate", CARRIAGE)
    xml = insert_after(xml, "js_knob", HANDLE)
    xml = insert_after(xml, "column", COLUMN)
    xml = insert_after(xml, "eyepiece_R", MICROSCOPE)
    xml = insert_after(xml, "forehead", HEADREST)
    return xml


def main():
    OUT.mkdir(exist_ok=True)
    for side, tag in ((1, "R"), (-1, "L")):
        dial_texture(side).save(OUT / f"dial_{tag}.png")
        (OUT / f"dial_{tag}.obj").write_text(dial_mesh(side))
    (MODELS / "scene_sl130.xml").write_text(build_scene(), encoding="utf-8")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODELS / "scene_sl130.xml"))
    print("saved models/scene_sl130.xml", f"bodies={m.nbody} joints={m.njnt} geoms={m.ngeom}")


if __name__ == "__main__":
    main()
