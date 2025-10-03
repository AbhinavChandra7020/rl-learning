import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py

gym.register_envs(ale_py)

env= gym.make("ALE/Breakout-v5")

print(env.reset())
print(env.action_space)
print(env.observation_space)