import numpy as np
from stable_baselines3 import DQN
import os
from grids.maze_grids import maze_grid_9x9
from environment import MazeEnv

log_path = os.path.join('logs')
save_path = os.path.join('saved_models', 'dqn_maze_solver9x9')

start_pos = [1,0]
goal_pos = [8,8]

env = MazeEnv(maze_grid_9x9, start_pos, goal_pos)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path,
            learning_rate=0.0005,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            target_update_interval=500,
            train_freq=4)

model.learn(total_timesteps=200000)

model.save(save_path)