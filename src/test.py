import time
import mujoco
import mujoco.viewer
import numpy as np

# Загружаем модель
model = mujoco.MjModel.from_xml_path("/home/nurzada/quadruped-assembler/biped-wheeled-robot/biped_wheeled_leg/biped_wheeled_leg1.xml")
data = mujoco.MjData(model)

# Задаем начальную позу (чтобы робот не упал сразу)
# Немного сгибаем колени и поднимаем туловище
data.qpos[2] = 0.5  # высота туловища (если есть freejoint)

# Запускаем просмотр
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(10000):
        # Опционально: подаем команды на моторы
        data.ctrl[0] = 0.5  # hip
        data.ctrl[1] = 0.3  # knee
        mujoco.mj_step(model, data)
        viewer.sync()