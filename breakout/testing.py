import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import ale_py

DQN_PATH = os.path.join('Training', 'Saved Models', 'dqn_breakout_final')

gym.register_envs(ale_py)

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=1,
                     env_kwargs={'render_mode': 'human'})
env = VecFrameStack(env, n_stack=4)

model = DQN.load(DQN_PATH)

episodes = 5
max_steps_per_episode = 500

for episode in range(1, episodes+1):
    obs = env.reset()  
    done = False
    score = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        score += reward[0]
        done = done[0]
        steps += 1
        
        if steps % 100 == 0:
            print(f'Episode {episode}, Steps: {steps}, Score: {score}')
    
    print(f'Episode {episode} finished - Final Score: {score}, Steps: {steps}')

env.close()