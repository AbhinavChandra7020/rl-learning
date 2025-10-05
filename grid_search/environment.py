import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_DELTAS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1)    # RIGHT
]

# define valid actions and positions to move to
def move(grid, current_pos, action):
    row, col = current_pos
    delta_row, delta_col = ACTION_DELTAS[action]

    new_row = row + delta_row
    new_col = col + delta_col
    
    if(0 <= new_row < grid.shape[0] and  0 <= new_col < grid.shape[1] and grid[new_row, new_col] == 1 ):
        return [new_row, new_col]
    else:
        return current_pos
    
# custom gym environment
class MazeEnv(gym.Env):
    def __init__(self, grid, start_pos, goal_pos):
        super().__init__()

        self.grid = grid
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos.copy()
        self.steps = 0  # Added step counter

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low = 0,
            high = grid.shape[0] - 1,
            shape = (2, ),
            dtype = np.int32
        )

    # reset the environment for each iteration
    def reset(self, seed=None, options=None):
        super().reset(seed = seed)

        self.current_pos = self.start_pos.copy()
        self.steps = 0  # Reset step counter

        observation = np.array(self.current_pos, dtype=np.int32)
        info = {}
        return observation, info
    
    # letting the agent move
    def step(self, action):
        old_distance = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        
        new_pos = move(self.grid, self.current_pos, action)
        moved = not np.array_equal(new_pos, self.current_pos)
        self.current_pos = new_pos
        self.steps += 1
        
        new_distance = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        reached_goal = np.array_equal(self.current_pos, self.goal_pos)
        
        if reached_goal:
            reward = 100
            terminated = True
        elif not moved:
            reward = -1
            terminated = False
        else:
            distance_reward = (old_distance - new_distance) * 1.0
            reward = -0.01 + distance_reward
            terminated = False
        
        if self.steps >= 200:
            terminated = True
        
        observation = np.array(self.current_pos, dtype=np.int32)
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info