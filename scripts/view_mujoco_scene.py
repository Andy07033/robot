#!/usr/bin/python3

from pathlib import Path
import time
import sys

import mujoco
import mujoco.viewer


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_xml = repo_root / "mujoco" / "mobile_dual_ur3_scene.xml"
    xml_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_xml
    if not xml_path.is_absolute():
        xml_path = (Path.cwd() / xml_path).resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        home_ctrl = model.key_ctrl[home_key].copy()
    else:
        home_ctrl = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.65, 0.75]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = -135
        viewer.cam.elevation = -22

        while viewer.is_running():
            # Hold position actuators at their home targets, keeping the scene calm
            # while still letting the plug free body and contacts settle naturally.
            if home_ctrl is not None:
                data.ctrl[:] = home_ctrl
            else:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
