import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import ale_py

A2C_PATH = os.path.join('Training', 'Saved Models', 'A2C_Breakout_Model')

gym.register_envs(ale_py)

model = A2C.load(A2C_PATH)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=1,
                          env_kwargs={'render_mode': 'human'})
eval_env = VecFrameStack(eval_env, n_stack=4)

for episode in range(5):
    obs = eval_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward[0]
        done = done[0]
    
    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

eval_env.close()