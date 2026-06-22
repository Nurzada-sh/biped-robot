import numpy as np
from pathlib import Path

XML_PATH   = "robot.xml"
DT         = 0.002
TAU_MAX    = 18.0
MAX_STEPS  = 6000
Q_STAND    = np.array([0.0, 0.3, 0.0, 0.3])
TARGET_VX  = 0.4

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
