import sys
import os
import numpy as np
import mujoco.viewer

sys.path.append(os.path.dirname(__file__))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.biped_env import BipedWheeledRobotEnv

XML_PATH = "/home/nurzada/quadruped-assembler/biped-wheeled-robot/biped_wheeled_leg/biped_wheeled_leg1.xml"

def test_sac(model_path, n_episodes=3):
    print(f"Загрузка SAC модели: {model_path}")
    
    model = SAC.load(model_path)
    
    for ep in range(n_episodes):
        print(f"\nЭпизод {ep+1}")
        
       
        env = BipedWheeledRobotEnv(XML_PATH)
        obs, _ = env.reset()
        
        total_reward = 0
        step = 0
        
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while True:
                
                action, _ = model.predict(obs, deterministic=True)
                
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                
                viewer.sync()
                
                if terminated:
                    print(f"  Падение на шаге {step}")
                    break
                if truncated:
                    print(f"  Эпизод завершён на шаге {step}")
                    break
            
            print(f"  Общая награда: {total_reward:.2f}")
            print(f"  Скорость: {info['forward_vel']:.2f} м/с")
        
        env.close()

def test_ppo(model_path, n_episodes=3):
    print(f"Загрузка PPO модели: {model_path}")
    model = PPO.load(model_path)
    
    for ep in range(n_episodes):
        print(f"\nЭпизод {ep+1}")
        env = BipedWheeledRobotEnv(XML_PATH)
        obs, _ = env.reset()
        
        total_reward = 0
        step = 0
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                viewer.sync()
                
                if terminated or truncated:
                    break
            
            print(f"  Шагов: {step}, Награда: {total_reward:.2f}, Скорость: {info['forward_vel']:.2f}")
        
        env.close()

if __name__ == "__main__":
    
    sac_path = "./models/sac_biped_final.zip"
    ppo_path = "./models/ppo_biped.zip"
    
    
    if os.path.exists(sac_path):
        test_sac(sac_path)
    elif os.path.exists(ppo_path):
        test_ppo(ppo_path)
    else:
        print("Модель не найдена! Сначала обучи: python train_sac.py")
