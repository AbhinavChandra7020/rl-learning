import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

log_path = os.path.join('Training', 'Logs')
DQN_PATH = os.path.join('Training' , 'Saved Models', 'DQN_Model_CartPole')

env = gym.make('CartPole-v1')

env = DummyVecEnv([lambda: env])

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

model.save(DQN_PATH)