import numpy as np
import matplotlib.pyplot as plt
from grids.maze_grids import maze_grid_9x9

def visualize_grid_with_agent(grid, agent_pos):
    
    plt.figure(figsize=(8, 8))
    
    plt.imshow(grid, cmap="gray", interpolation='nearest')

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)

    ax.grid(which='minor', color='lightblue', linestyle='-', linewidth=1)

    plt.scatter(agent_pos[1],agent_pos[0], s=300, c='red', zorder=10)

    plt.title('9x9 Maze with Agent')

    plt.savefig('visualized_grids/9x9_grid.png') 
    plt.show()

agent_start = [1, 0] 
visualize_grid_with_agent(maze_grid_9x9, agent_start)