import os
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy 

log_path = os.path.join('Training', 'Logs')
PPO_PATH = os.path.join('Training' , 'Saved Models', 'PPO_Model_CartPole')

# Create environment
env = gym.make('CartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")