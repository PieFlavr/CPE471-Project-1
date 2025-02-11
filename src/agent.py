"""
agent.py

Description: This module defines an Agent class that can move within a grid.
Author: Lucas Pinto
Date: February 10, 2025

Modules:
    None

Classes:
    Agent

Functions:
    None

Usage:
    agent = Agent((0, 0))
    agent.move('up', (5, 5))
"""

from typing import Tuple
from utils import Position

class Agent:

    def __init__(self, position: Tuple[int,int] = (0,0)):
        """
        Initialize the agent with a starting position.

        Args:
            position (Tuple[int,int], optional): The initial position of the agent as (x, y). Defaults to (0,0).
        """
        self.position = Position(position[0], position[1])

    def move(self, action: str = None, grid_dim: Tuple[int,int] = (5,5)):
        """
        Move the agent in the specified direction within the grid dimensions.

        Args:
            action (str, optional): The direction in which to move the agent. 
                        Can be 'up', 'down', 'left', or 'right'. Defaults to None.
            grid_dim (Tuple[int,int], optional): The dimensions of the grid as (length, width). 
                             Defaults to (5,5).

        Returns:
            Tuple[int, int]: The new position of the agent as (x, y).
        """
        x, y = self.position.x, self.position.y
        grid_length, grid_width = grid_dim[0], grid_dim[1]

        if action == 'up' and x > 0:
            self.position.x -= 1
        elif action == 'down' and x < grid_length - 1:
            self.position.x += 1
        elif action == 'left' and y > 0:
            self.position.y -= 1
        elif action == 'right' and y < grid_width - 1:
           self.position.y += 1
        else:
            pass
        return Tuple(self.position.x,self.position.y)