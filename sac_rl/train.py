import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
import os

class BipedWheeledRobotEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        
        self.action_space = spaces.Box(low=-30, high=30, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        self.joint_names = ['rhip_pitch_joint', 'rknee_crank_joint', 'lhip_pitch_joint', 'lknee_crank_joint']
        self.joint_qpos_addr = []
        self.joint_qvel_addr = []
        for name in self.joint_names:
            jid = self.model.joint(name).id
            self.joint_qpos_addr.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_addr.append(self.model.jnt_dofadr[jid])
    
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.4
        self.data.qpos[3] = 1.0
        self.data.qpos[7] = 0.3
        self.data.qpos[8] = -0.2
        self.data.qpos[9] = -0.3
        self.data.qpos[10] = 0.2
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.data.body('torso').xpos[2] < 0.2
        return obs, reward, terminated, False, {}
    
    def _get_obs(self):
        obs = []
        for addr in self.joint_qpos_addr:
            obs.append(self.data.qpos[addr])
        for addr in self.joint_qvel_addr:
            obs.append(self.data.qvel[addr])
        obs.extend(self.data.body('torso').xpos)
        obs.extend(self.data.body('torso').xquat)
        obs.extend(self.data.body('torso').cvel)
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self):
        forward_vel = self.data.body('torso').cvel[0]
        height = self.data.body('torso').xpos[2]
        energy = np.sum(np.square(self.data.ctrl)) * 0.0001
        return forward_vel + (0.05 if height > 0.25 else 0) - energy

if __name__ == "__main__":
    XML_PATH = "/home/nurzada/quadruped-assembler/biped-wheeled-robot/biped_wheeled_leg/biped_wheeled_leg1.xml"
    
    env = BipedWheeledRobotEnv(XML_PATH)
    
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=200_000)  
    model.save("sac_biped_model")
    
    print("✅ Обучение завершено!")
    env.close()
