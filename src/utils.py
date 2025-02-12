"""
utils.py

Description: Utility functions for the project.
Author: Lucas Pinto
Date: February 10, 2025

Modules:
    None

Functions:
    get_key_by_value

Usage:
"""
import numpy as np
import matplotlib.pyplot as plt

def get_key_by_value(dictionary, target_value):
    """
    Retrieves the key associated with the given value in a dictionary.

    Args:
        dictionary (dict): The dictionary to search through.
        target_value (any): The value to find the corresponding key for.

    Returns:
        any: The key associated with the target value, or None if the value is not found.
    """
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

def q_table_to_2d_array(q_table, grid_length, grid_width):
    """
    Converts a Q-table into a 2D array representation.

    Args:
        q_table (np.ndarray): The Q-table to convert, assumed to be a 3D array with shape (grid_length, grid_width, num_actions).
        grid_length (int): The length of the grid.
        grid_width (int): The width of the grid.

    Returns:
        np.ndarray: A 2D array where each row represents a state and its corresponding Q-values.
    """
    rows = []
    for x in range(grid_length):
        for y in range(grid_width):
            row = [f"({x},{y})"] + list(q_table[x, y, :])
            rows.append(row)
    return np.array(rows)

def plot_action_sequence(action_sequence, grid_length, grid_width, title, subtitle=None):
    """
    Plots the action sequence on a grid.

    Args:
        action_sequence (list): List of actions taken by the agent.
        grid_length (int): Length of the grid.
        grid_width (int): Width of the grid.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0,grid_width)
    ax.set_ylim(0,grid_length)
    ax.set_xticks(np.arange(0, grid_width, 1))
    ax.set_yticks(np.arange(0, grid_length, 1))
    ax.grid(True)

    # Initial position
    x, y = 0.5, 0.5
    dx, dy = 0, 0
    print(action_sequence)
    for action in action_sequence:
        if action == 0: # Up
            dx, dy = 0, -1
        elif action == 1: # Down
            dx, dy = 0, 1
        elif action == 2: # Left
            dx, dy = -1, 0
        elif action == 3: # Right
            dx, dy = 1, 0

        ax.arrow(x, y, dx*0.25, dy*0.25, head_width=0.25, head_length=0.25, fc='blue', ec='blue')
        # Update position with validation
        new_x = x + dx
        new_y = y + dy

        # Ensure the new position is within grid boundaries
        if (0.5 <= new_x < grid_length+0.5) and (0.5 <= new_y < grid_width+0.5):
            x, y = new_x, new_y
        else:
            print(f"Invalid move to ({new_x}, {new_y}) ignored.")

    plt.title(title)
    plt.suptitle(subtitle, fontsize=8)
    plt.gca().invert_yaxis()
    plt.show()