"""M0: シーンを読み込み、俯瞰カメラと細隙灯カメラの静止画を保存する。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/render_snapshot.py
"""
from pathlib import Path

import imageio.v3 as iio
import mujoco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def set_joint(model, data, name, value):
    """関節名を指定して位置（qpos）を書き込む。"""
    data.qpos[model.joint(name).qposadr[0]] = value


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    print(f"ボディ数={model.nbody}, 関節数={model.njnt}, 形状数={model.ngeom}, カメラ数={model.ncam}")

    renderer = mujoco.Renderer(model, height=480, width=640)

    # 2つの状態を撮る：中央（両眼の間）と、患者の右眼に合わせた状態
    poses = {"center": 0.0, "right_eye": -0.032}
    for tag, lamp_x in poses.items():
        set_joint(model, data, "lamp_x", lamp_x)
        mujoco.mj_forward(model, data)  # 位置を書き換えたら、各部品の位置を再計算する

        for cam in ("overview", "slit_cam"):
            renderer.update_scene(data, camera=cam)
            path = OUT / f"m0_{tag}_{cam}.png"
            iio.imwrite(path, renderer.render())
            print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
