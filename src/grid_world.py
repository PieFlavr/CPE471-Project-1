import numpy as np
from typing import Tuple

from agent import Agent

class GridWorld:
    def __init__(self, grid_dim: Tuple[int,int] = (5, 5), agent: Agent = None, goal: Tuple[int,int] = None):
        """
        Initialize the GridWorld with dimensions, agent, and goal.

        Args:
            grid_dim (tuple, optional): Dimensions of the grid as (rows, columns). Defaults to (5, 5).
            agent (Agent, optional): An instance of the Agent class. Defaults to None.
            goal (tuple, optional): Coordinates of the goal position as (x, y). Defaults to (4, 4).
        """
        self._grid_dim = grid_dim
        self._agent = agent if agent is not None else Agent() # If no agent is provided, create a new one
        self._goal = goal if goal is not None else (self._grid_dim[0] - 1, self._grid_dim[1] - 1) # If no goal is provided, set it to the bottom-right corner
        self._grid = np.zeros(grid_dim)
        self._grid[agent.position.x, agent.position.y] = 2 # Represents the agent as a 2 in the grid

    def move_agent(self, action: str = None) -> bool:
        """
        Move the agent in the specified direction and update the grid.

        Args:
            action (str, optional): The direction to move the agent. Can be 'up', 'down', 'left', or 'right'. Defaults to None.

        Returns:
            bool: True if the action was successful, False otherwise.
        """
        self._grid[self._agent.position.x, self._agent.position.y] = 0
        self._agent.move(action, self._grid_dim)
        self._grid[self._agent.position.x, self._agent.position.y] = 2
    
    def reset_agent(self):
        """
        Reset the agent's position to a random position within the grid excluding the goal.
        """
        self._grid = np.zeros(self._grid_dim, dtype = int)
        while True:
            self._agent.position = (np.random.choice(self._grid_dim[0]), np.random.choice(self._grid_dim[1]))
            if(any(self._agent.position != self._goal)):
                break
        self._grid[self._agent.position.x, self._agent.position.y] = 2
    
    def get_agent_position(self):
        """
        Get the current position of the agent.

        Returns:
            tuple: The (x, y) coordinates of the agent's current position.
        """
        return tuple(self._agent.position.x, self._agent.position.y)
    