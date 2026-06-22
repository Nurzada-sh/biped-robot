import os
import math
import numpy as np
import mujoco
import pickle
import gymnasium as gym
from gymnasium import spaces
from config import XML_PATH, DT, TAU_MAX, MAX_STEPS, Q_STAND, TARGET_VX


class DREMOffline:
    def __init__(self, n_params=4, gamma=0.95, lam=0.5, reg_param=0.005):
        self.n = n_params
        self.gamma = gamma
        self.lam = lam
        self.reg_param = reg_param
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.n)
        self.phi_f = np.zeros(self.n)
        self.y_f = 0.0
        self.Omega = np.eye(self.n) * 0.1
        self.G = np.zeros(self.n)
        self.count = 0

    def step(self, phi, y, dt):
        alpha = np.exp(-self.lam * dt)
        self.phi_f = alpha * self.phi_f + (1 - alpha) * phi
        self.y_f = alpha * self.y_f + (1 - alpha) * y
        self.Omega += self.gamma * np.outer(self.phi_f, self.phi_f) * dt
        self.G += self.gamma * self.phi_f * self.y_f * dt
        self.count += 1
        try:
            Omega_reg = self.Omega + self.reg_param * np.eye(self.n)
            theta_new = np.linalg.solve(Omega_reg, self.G)
            alpha_th = 0.01 if self.count < 100 else 0.05
            self.theta = (1 - alpha_th) * self.theta + alpha_th * theta_new
        except Exception:
            pass
        return self.theta

    def fit(self, phi_hist, y_hist, dt):
        self.reset()
        for i in range(len(y_hist)):
            self.step(phi_hist[i], y_hist[i], dt)
        return self.theta


class DREMActuator:
    DEFAULT_PARAMS = np.array([
        [2.0, 0.08, 0.15, 0.008],
        [2.5, 0.10, 0.12, 0.010],
        [2.0, 0.08, 0.15, 0.008],
        [2.5, 0.10, 0.12, 0.010],
    ])

    def __init__(self, n_act=4, dt=DT):
        self.n_act = n_act
        self.dt = dt
        self.theta = self.DEFAULT_PARAMS.copy()
        self.dq_prev = np.zeros(n_act)

    def load_params(self, params: np.ndarray):
        self.theta = np.array(params, dtype=float)

    @property
    def theta_flat(self) -> np.ndarray:
        return self.theta.flatten()

    def inverse(self, tau_des: np.ndarray, dq: np.ndarray) -> np.ndarray:
        u_cmd = np.zeros(self.n_act)
        for i in range(self.n_act):
            Kt, B, Fc, J = self.theta[i]
            ddq = (dq[i] - self.dq_prev[i]) / self.dt
            Fc_sign = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0.0
            u_cmd[i] = (tau_des[i] + B * dq[i] + Fc * Fc_sign + J * ddq) / (Kt + 1e-6)
        self.dq_prev = dq.copy()
        return np.clip(u_cmd, -1.0, 1.0)

    def forward(self, u_cmd: np.ndarray, dq: np.ndarray) -> np.ndarray:
        tau_real = np.zeros(self.n_act)
        for i in range(self.n_act):
            Kt, B, Fc, J = self.theta[i]
            ddq = (dq[i] - self.dq_prev[i]) / self.dt
            Fc_sign = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0.0
            tau_real[i] = Kt * u_cmd[i] - B * dq[i] - Fc * Fc_sign - J * ddq
        self.dq_prev = dq.copy()
        return tau_real


class BipedalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    JOINT_NAMES = ["L_hip", "L_knee", "R_hip", "R_knee"]
    MOTOR_NAMES = ["L_hip_m", "L_knee_m", "R_hip_m", "R_knee_m"]
    WHEEL_NAMES = ["L_wheel_m", "R_wheel_m"]

    def __init__(self, render_mode=None, use_drem_params: bool = True):
        super().__init__()
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"MuJoCo model not found: {XML_PATH}")
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.mj_data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.use_drem = use_drem_params

        self.jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.JOINT_NAMES]
        self.mids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.MOTOR_NAMES]
        self.wheel_mids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.WHEEL_NAMES]

        self._foot_geom_ids = []
        for name in ["L_foot", "R_foot", "foot_L", "foot_R"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._foot_geom_ids.append(gid)

        self.actuator = DREMActuator(n_act=4, dt=DT)
        if use_drem_params:
            self._load_drem()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 6 + 1 + 2 + (16 if use_drem_params else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _load_drem(self):
        path = "data/drem_params.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.actuator.load_params(pickle.load(f))

    def _get_foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(2)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            geom = [c.geom1, c.geom2]
            for gid in self._foot_geom_ids:
                if gid in geom:
                    contacts[gid % 2] = 1.0
        return contacts

    def _get_joint_state(self):
        q = np.array([
            self.mj_data.qpos[7],
            self.mj_data.qpos[8],
            self.mj_data.qpos[10],
            self.mj_data.qpos[11],
        ])
        dq = np.array([self.mj_data.qvel[self.model.jnt_dofadr[jid]] for jid in self.jids])
        return q, dq

    def _get_imu(self):
        qw, qx, qy, qz = self.mj_data.qpos[3:7]
        roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = math.asin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        omega = self.mj_data.qvel[3:6].copy()
        return np.array([roll, pitch, yaw, *omega])

    def _get_obs(self) -> np.ndarray:
        q, dq = self._get_joint_state()
        imu = self._get_imu()
        vx = np.array([self.mj_data.qvel[0]])
        foot = self._get_foot_contacts()
        if self.use_drem:
            obs = np.concatenate([q, dq, imu, vx, foot, self.actuator.theta_flat])
        else:
            obs = np.concatenate([q, dq, imu, vx, foot])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.mj_data)
        self.step_count = 0
        self.actuator = DREMActuator(n_act=4, dt=DT)
        if self.use_drem:
            self._load_drem()

        self.mj_data.qpos[2] = 0.72
        self.mj_data.qpos[3:7] = [1, 0, 0, 0]
        self.mj_data.qpos[7] = Q_STAND[0] + np.random.uniform(-0.05, 0.05)
        self.mj_data.qpos[8] = Q_STAND[1] + np.random.uniform(-0.05, 0.05)
        self.mj_data.qpos[10] = Q_STAND[2] + np.random.uniform(-0.05, 0.05)
        self.mj_data.qpos[11] = Q_STAND[3] + np.random.uniform(-0.05, 0.05)
        self.mj_data.qvel[:] = np.random.uniform(-0.02, 0.02, self.model.nv)
        mujoco.mj_forward(self.model, self.mj_data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        tau_des = np.array(action, dtype=float) * TAU_MAX
        _, dq = self._get_joint_state()
        u_cmd = self.actuator.inverse(tau_des, dq)
        tau_real = self.actuator.forward(u_cmd, dq)

        self.mj_data.ctrl[:] = 0.0
        for i in range(4):
            self.mj_data.ctrl[self.mids[i]] = np.clip(tau_real[i], -TAU_MAX, TAU_MAX)

        imu = self._get_imu()
        pitch = imu[1]
        pitch_rate = imu[4]
        wheel_tau = np.clip(30.0 * pitch + 3.0 * pitch_rate, -5.0, 5.0)
        for wid in self.wheel_mids:
            if wid >= 0:
                self.mj_data.ctrl[wid] = wheel_tau

        mujoco.mj_step(self.model, self.mj_data)
        self.step_count += 1
        reward, info = self._compute_reward(tau_des, imu)
        roll, pitch = imu[0], imu[1]
        h = self.mj_data.qpos[2]
        terminated = bool(abs(pitch) > 1.2 or abs(roll) > 1.2 or h < 0.30)
        truncated = bool(self.step_count >= MAX_STEPS)
        return self._get_obs(), float(reward), terminated, truncated, info

    def _compute_reward(self, tau_des, imu):
        roll, pitch = imu[0], imu[1]
        vx = self.mj_data.qvel[0]
        h = self.mj_data.qpos[2]
        q, _ = self._get_joint_state()

        r_velocity = 2.0 * np.exp(-4.0 * (vx - TARGET_VX) ** 2)
        r_alive = 0.5
        r_upright = 1.0 * np.exp(-6.0 * pitch ** 2) + 0.5 * np.exp(-6.0 * roll ** 2)
        r_height = 0.3 * np.exp(-20.0 * (h - 0.72) ** 2)
        p_energy = 0.003 * np.sum(np.abs(tau_des))
        q_limit = np.array([-0.8, 0.0, -0.8, 0.0])
        q_upper = np.array([0.8, 1.2, 0.8, 1.2])
        p_joint_lim = 0.2 * np.sum(np.maximum(0, q - q_upper) ** 2 + np.maximum(0, q_limit - q) ** 2)
        reward = r_velocity + r_alive + r_upright + r_height - p_energy - p_joint_lim
        reward = np.clip(reward, -2.0, 5.0)
        info = {
            "r_velocity": r_velocity,
            "r_alive": r_alive,
            "r_upright": r_upright,
            "r_height": r_height,
            "p_energy": p_energy,
            "p_joint_lim": p_joint_lim,
            "vx": float(vx),
            "pitch": float(pitch),
            "height": float(h),
        }
        return reward, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.mj_data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
