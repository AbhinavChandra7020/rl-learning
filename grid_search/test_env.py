
import numpy as np
from grids.maze_grids import maze_grid_9x9
from environment import MazeEnv

start_pos = [1, 0]
goal_pos = [8, 8]

env = MazeEnv(maze_grid_9x9, start_pos, goal_pos)

print("Environment created successfully")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

print("\n--- Testing Reset ---")
observation, info = env.reset()
print(f"Initial observation (agent position): {observation}")
print(f"Expected starting position: {start_pos}")
print(f"Match: {np.array_equal(observation, start_pos)}")

# Test stepping through the environment
print("\n--- Testing Step Function ---")

# Action 3 = RIGHT (should move from [1,0] to [1,1])
print(f"Current position: {env.current_pos}")
observation, reward, terminated, truncated, info = env.step(3)
print(f"Action: RIGHT (3)")
print(f"New position: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

# Action 0 = UP (might hit wall at [0,1] which is a wall)
print(f"\nCurrent position: {env.current_pos}")
observation, reward, terminated, truncated, info = env.step(0)
print(f"Action: UP (0)")
print(f"New position: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")

# Action 2 = LEFT (should hit boundary, stay at same position)
print(f"\nCurrent position: {env.current_pos}")
observation, reward, terminated, truncated, info = env.step(2)
print(f"Action: LEFT (2)")
print(f"New position: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")

# Test with visualization
print("\n--- Testing with Visualization ---")
from visualize_grid import visualize_grid_with_agent

# Reset environment
env.reset()
print("Starting position - press close to continue")
visualize_grid_with_agent(env.grid, env.current_pos)

# Move right
env.step(3)
print("After moving RIGHT - press close to continue")
visualize_grid_with_agent(env.grid, env.current_pos)

# Move right again
env.step(3)
print("After moving RIGHT again - press close to continue")
visualize_grid_with_agent(env.grid, env.current_pos)

env.step(3)
print("After moving RIGHT again - press close to continue")
visualize_grid_with_agent(env.grid, env.current_pos)

# Move down
env.step(1)
print("After moving DOWN - press close to continue")
visualize_grid_with_agent(env.grid, env.current_pos)