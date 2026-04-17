import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/nurzada/quadruped-assembler/IZ2/models/one_leg.xml/robot.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()