import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class BipedWheeledRobotEnv(gym.Env):
    def __init__(self, xml_path: str, render_mode: Optional[str] = None):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        
       
        self.action_space = spaces.Box(low=-30, high=30, shape=(4,), dtype=np.float32)
        
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        self.joint_names = ['rhip_pitch_joint', 'rknee_crank_joint', 'lhip_pitch_joint', 'lknee_crank_joint']
        
        
        self.joint_qpos_addr = []
        self.joint_qvel_addr = []
        for name in self.joint_names:
            joint_id = self.model.joint(name).id
            self.joint_qpos_addr.append(self.model.jnt_qposadr[joint_id])
            self.joint_qvel_addr.append(self.model.jnt_dofadr[joint_id])
        
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        
        self.data.qpos[2] = 0.4   
        self.data.qpos[3] = 1.0   
        self.data.qpos[7] = 0.3   
        self.data.qpos[8] = -0.2  
        self.data.qpos[9] = -0.3  
        self.data.qpos[10] = 0.2  
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        
        self.data.ctrl[:] = action
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        
        torso_height = self.data.body('torso').xpos[2]
        terminated = torso_height < 0.2
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        info = {
            'height': torso_height,
            'forward_vel': self.data.body('torso').cvel[0]
        }
        
        return obs, reward, terminated, truncated, info
    
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
        height_reward = 0.05 if height > 0.25 else 0
        return forward_vel + height_reward - energy
    
    def render(self):
        if self.render_mode == 'human' and self.viewer is None:
            mujoco.viewer.launch_passive(self.model, self.data)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
