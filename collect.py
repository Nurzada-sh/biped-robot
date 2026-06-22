import os
import numpy as np
import mujoco
import pickle
from scipy.signal import butter, lfilter
from config import XML_PATH


def collect_and_identify():
    if not os.path.exists(XML_PATH):
        return
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    joint_names = ["L_hip", "L_knee", "R_hip", "R_knee"]
    motor_names = ["L_hip_m", "L_knee_m", "R_hip_m", "R_knee_m"]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    mids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in motor_names]
    duration = 25.0
    f0, f1 = 0.3, 5.0
    dt = model.opt.timestep
    t_vec = np.arange(0, duration, dt)
    log = {'t': [], 'u': [], 'q': [], 'dq': [], 'tau': []}
    for step, t in enumerate(t_vec):
        k = (f1 - f0) / (2 * duration)
        phase_hip = 2 * np.pi * (f0 * t + k * t * t)
        phase_kne = phase_hip + np.pi / 4
        u_hip = 6.0 * (np.sin(phase_hip) + 0.2 * np.sin(2 * phase_hip + 0.5))
        u_kne = 5.0 * (np.sin(phase_kne) + 0.2 * np.sin(2 * phase_kne + 0.5))
        data.ctrl[mids[0]] = u_hip
        data.ctrl[mids[2]] = u_hip
        data.ctrl[mids[1]] = u_kne
        data.ctrl[mids[3]] = u_kne
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            log['t'].append(t)
            log['u'].append([u_hip, u_kne, u_hip, u_kne])
            log['q'].append([data.qpos[7], data.qpos[8], data.qpos[10], data.qpos[11]])
            log['dq'].append([data.qvel[model.jnt_dofadr[jid]] for jid in jids])
            log['tau'].append([data.actuator_force[mid] for mid in mids])
    for k in log:
        log[k] = np.array(log[k])
    with open("data/excitation.pkl", "wb") as f:
        pickle.dump(log, f)
    u, dq, tau = log['u'], log['dq'], log['tau']
    Ts = log['t'][1] - log['t'][0]
    b, a = butter(3, 25.0, btype='low', fs=1.0/Ts)
    dq_f = lfilter(b, a, dq, axis=0)
    tau_f = lfilter(b, a, tau, axis=0)
    ddq = np.gradient(dq_f, Ts, axis=0)
    new_params = []
    for i in range(4):
        X = np.column_stack([u[:, i], -dq_f[:, i], -np.sign(dq_f[:, i]), -ddq[:, i]])
        Y = tau_f[:, i]
        theta = np.abs(np.linalg.lstsq(X, Y, rcond=None)[0])
        new_params.append(theta)
    with open("data/drem_params.pkl", "wb") as f:
        pickle.dump(np.array(new_params), f)
