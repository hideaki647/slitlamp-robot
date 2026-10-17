"""M1 Step3：IK でアームの手をノブに合わせ、握った状態の静止画を撮る。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m1_step3_ik.py
"""
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from sim.slitlamp_api import SlitLampRobot
from sim.viz import free_camera, label, side_camera

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    robot = SlitLampRobot(model, data)

    e_pos, e_rot = robot.grasp()
    print(f"IK誤差: 位置 {e_pos * 1000:.3f} mm, 角度 {np.rad2deg(e_rot):.3f} deg")
    print("アーム関節角[deg]:", np.round(np.rad2deg(robot.q_arm), 1))

    # 1秒間そのまま物理を進めて、握った姿勢が保たれるか（ずれ・振動がないか）を見る
    for _ in range(500):
        mujoco.mj_step(model, data)
    print(f"1秒後の手とノブのずれ: {robot.weld_error() * 1000:.3f} mm")

    renderer = mujoco.Renderer(model, height=480, width=640)
    cams = {
        "side": side_camera(),
        "overview": "overview",
        "closeup": free_camera(data.site_xpos[model.site("js_grip").id], 0.35, 45, -25),
    }
    images = []
    for name, cam in cams.items():
        renderer.update_scene(data, camera=cam)
        images.append(label(renderer.render(), [name]))
    path = OUT / "m1_step3_grasp.png"
    imageio.imwrite(path, np.hstack(images))
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
