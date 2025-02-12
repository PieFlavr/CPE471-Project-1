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
import matplotlib.animation as animation

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

def plot_action_sequence(action_sequence, grid_length, grid_width, title, subtitle=None, fps=48):
    """
    Plots the action sequence on a grid with a gradient effect.

    Args:
        action_sequence (list): List of actions taken by the agent.
        grid_length (int): Length of the grid.
        grid_width (int): Width of the grid.
        title (str): Title of the plot.
        subtitle (str, optional): Subtitle of the plot.
        fps (int, optional): Frames per second for the animation. Default is 48.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, grid_length)
    ax.set_ylim(0, grid_width)
    ax.set_xticks(np.arange(0, grid_length, 1))
    ax.set_yticks(np.arange(0, grid_width, 1))
    ax.grid(True)

    # Initial position
    x, y = 0.5, 0.5
    dx, dy = 0, 0
    cmap = plt.get_cmap('inferno')  # Colormap for gradient effect
    num_actions = len(action_sequence)
    base_fps = 60

    # Highlight the start point
    ax.plot(x, y, 'go', markersize=10, label='Start')
    ax.plot(x + grid_length - 1, y + grid_width - 1, 'ro', markersize=10, label='Goal')

    def update(frame):
        nonlocal x, y, dx, dy
        actions_per_frame = max(1, int(fps / base_fps))  # Adjust this value to control how many actions are processed per frame
        start_frame = frame * actions_per_frame
        end_frame = min(start_frame + actions_per_frame, num_actions)

        for i in range(start_frame, end_frame):
            action = action_sequence[i]
            if action == 0:  # Up
                dx, dy = 0, -1
            elif action == 1:  # Down
                dx, dy = 0, 1
            elif action == 2:  # Left
                dx, dy = -1, 0
            elif action == 3:  # Right
                dx, dy = 1, 0

            color = cmap(i / num_actions)  # Get color from colormap
            ax.arrow(x, y, dx * 0.75, dy * 0.75, head_width=0.25, head_length=0.25, fc=color, ec=color)

            # Update position with validation
            new_x = x + dx
            new_y = y + dy

            # Ensure the new position is within grid boundaries
            if (0.5 <= new_x < grid_length + 0.5) and (0.5 <= new_y < grid_width + 0.5):
                x, y = new_x, new_y
                #print(f"Frame {i}: Successful draw arrow draw from ({x - dx}, {y - dy}) to ({x}, {y}).")
            else:
                ax.arrow(x, y, dx * 0.25, dy * 0.25, head_width=0.25, head_length=0.25, fc='red', ec='red')
                #print(f"Frame {i}: Invalid move to ({new_x}, {new_y}) ignored.")

    print("Generating action sequence plot...")

    if fps is not 0:
        interval = 1000 / fps  # Calculate interval in milliseconds
        num_frames = (num_actions + max(1, int(fps / base_fps)) - 1) // max(1, int(fps / base_fps))  # Calculate the number of frames needed
        print(f"Animating action sequence with {num_actions} actions at {fps} FPS or {interval} ms interval.")
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=False)
    else:
        for i in range(num_actions):
            update(i)
            
    print("Action sequence plot complete.")

    plt.title(title)
    plt.suptitle(subtitle, fontsize=8)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()