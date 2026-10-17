"""動画づくりの共通部品：カメラの用意、画像への文字入れ、2画面の合成。"""
import cv2
import mujoco
import numpy as np


def free_camera(lookat, distance, azimuth, elevation):
    """注視点・距離・方位角・仰角で決める自由カメラ。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def side_camera():
    """真横（検者から見て左側）から、細隙灯・アーム・患者を見るカメラ。"""
    return free_camera([0.0, -0.10, 0.95], 1.15, 0, -15)


def top_camera():
    """真上から見下ろすカメラ。画面の上が患者側(+y)、下がアーム（医師の席）側。"""
    return free_camera([0.0, -0.03, 0.85], 1.4, 90, -89.9)


def label(img, lines):
    """画像の左上に英字の説明を書く（OpenCV の標準フォントは日本語非対応）。"""
    img = img.copy()
    for i, text in enumerate(lines):
        y = 28 + 26 * i
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def two_panel(renderer, data, left_cam, left_lines, right_lines):
    """左：指定カメラ、右：細隙灯カメラ を横に並べた1枚を返す。"""
    renderer.update_scene(data, camera=left_cam)
    left = label(renderer.render(), left_lines)
    renderer.update_scene(data, camera="slit_cam")
    right = label(renderer.render(), right_lines)
    return np.hstack([left, right])


def smooth(a, b, s):
    """a から b へなめらかに変化させる（s は 0〜1、始めと終わりがゆっくり）。"""
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s
