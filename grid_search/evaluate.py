import numpy as np
from stable_baselines3 import DQN
from grids.maze_grids import maze_grid_9x9
from environment import MazeEnv
from visualize_grid import visualize_grid_with_agent
import time

model = DQN.load("saved_models/dqn_maze_solver9x9")

start_pos = [1, 0]
goal_pos = [8, 8]
env = MazeEnv(maze_grid_9x9, start_pos, goal_pos)

for episode in range(5):
    print(f"\n=== Episode {episode + 1} ===")
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # visualize_grid_with_agent(env.grid, obs)
        # time.sleep(0.2)
    
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Reached goal: {done}")
    print(f"Final position: {obs}")

print("\n=== Summary ===")
print("Training complete! Run with visualization uncommented to watch it solve the maze.")