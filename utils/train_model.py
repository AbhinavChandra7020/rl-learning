import os
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv 

timesteps = 100000

log_path = os.path.join('Training', 'Logs')
PPO_PATH = os.path.join('Training' , 'Saved Models', 'PPO_Model_CartPole')

env = gym.make('CartPole-v1')

env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=timesteps)

model.save(PPO_PATH)