import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from core import BipedalEnv
from config import MAX_STEPS


class TrainingProgressCallback(BaseCallback):
    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.ep_rewards = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self._ep_reward += reward
        if done:
            self.ep_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        if self.num_timesteps % self.log_interval == 0 and self.ep_rewards:
            mean_r = np.mean(self.ep_rewards[-20:])
            print(f"  [{self.num_timesteps:>8} steps] mean_reward={mean_r:>7.1f}")
        return True


def train_sac(use_drem: bool = True, num_steps: int = 300_000):
    tag = "with_drem" if use_drem else "without_drem"
    def make_env():
        def _init():
            env = BipedalEnv(render_mode=None, use_drem_params=use_drem)
            env = Monitor(env, filename=f"logs/monitor_{tag}")
            return env
        return _init
    train_env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])
    model = SAC("MlpPolicy", train_env, verbose=0,
                learning_rate=3e-4, buffer_size=500_000, learning_starts=10_000,
                batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                ent_coef="auto", target_entropy="auto",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]), log_std_init=-3))
    progress_cb = TrainingProgressCallback(log_interval=10_000)
    checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path=f"models/checkpoints_{tag}", name_prefix=f"sac_{tag}")
    eval_cb = EvalCallback(eval_env, best_model_save_path=f"models/best_{tag}", eval_freq=20_000,
                           n_eval_episodes=5, deterministic=True, verbose=1)
    model.learn(total_timesteps=num_steps, callback=[progress_cb, checkpoint_cb, eval_cb])
    model.save(f"models/sac_{tag}")
    train_env.close()
    eval_env.close()


def test_visual(model_type: str = "with_drem", n_episodes: int = 5):
    use_drem = "with" in model_type
    best_path = f"models/best_{model_type}/best_model"
    final_path = f"models/sac_{model_type}"
    if os.path.exists(best_path + ".zip"):
        load_path = best_path
    elif os.path.exists(final_path + ".zip"):
        load_path = final_path
    else:
        return
    env = BipedalEnv(render_mode="human", use_drem_params=use_drem)
    model = SAC.load(load_path, env=env)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            env.render()
            if terminated or truncated:
                break
    env.close()
