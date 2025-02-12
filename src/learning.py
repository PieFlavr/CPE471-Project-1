import numpy as np

from utils import *
from grid_world import GridWorld
from agent import Agent
from typing import Tuple

def Q_learning_episode(grid_world: GridWorld = None, 
               agent: Agent = None, 
               actions: list = None,
               q_table: np.ndarray = None,
               selection_function: callable = None,
               function_args: dict = None,
               alpha: float = 0.1, 
               gamma: float = 0.9, 
               agent_start: Tuple[int,int] = None) -> Tuple[list, float, int, list]:
    """
    Runs a single episode of the Q-learning algorithm.
    Returns a list of the training data per episode. 
    Per episode returns a list of, in order: action sequence, total reward, steps taken, and Q-table history.

    Args:
        grid_world (GridWorld, optional): The environment in which the agent operates. Defaults to None.
        agent (Agent, optional): The agent that interacts with the environment. Defaults to None.
        actions (list, optional): List of possible actions the agent can take. Defaults to None.
        q_table (np.ndarray, optional): Q-table used to store and update Q-values. Defaults to None.
        selection_function (callable, optional): Function used to select actions based on Q-values. Defaults to None.
        function_args (dict, optional): Arguments for the selection function. Defaults to None.
        alpha (float, optional): Learning rate for Q-learning updates. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        agent_start (Tuple[int, int], optional): Starting position of the agent. Defaults to None.

    Raises:
        ValueError: If any of the required parameters (grid_world, actions, q_table, selection_function) are None.
        ValueError: If selection_function is not callable or its arguments are invalid.

    Returns:
        Tuple[list, float, int, list]: A tuple containing:
            - action_sequence (list): Sequence of actions taken by the agent.
            - total_reward (float): Total reward accumulated during the episode.
            - steps_taken (int): Number of steps taken to reach the goal.
            - q_table_history (list): History of Q-table updates during the episode.
    """
    
    # Check if any of the parameters are None
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if q_table is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())

    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        selection_function(test_state, **function_args)
    except TypeError as e:
        raise ValueError(f"Selection function arguments are invalid: {e}")

    grid_world.reset(agent_start)  # Initializes the agent and environment state

    action_sequence = []
    q_table_history = []
    steps_taken = 0
    total_reward = 0

    goal_reached = False

    while not goal_reached:
        q_table_history.append(q_table.copy())

        state = grid_world.get_state()[1]  # Get the current state of the environment
        action = selection_function(state, **function_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action)
        steps_taken += 1
        total_reward += reward

        next_state = grid_world.get_state()[1]  # Get the next state of the environment

        Q_learning_table_upate(state, next_state, action, reward, q_table, alpha, gamma)

    return action_sequence, total_reward, steps_taken, q_table_history

def epsilon_greedy_selection(state: Tuple[int, ...], q_table: np.ndarray = None, epsilon: float = 0.1) -> int:
    """
    Selects an action using the epsilon-greedy policy.

    Args:
        state (Tuple[int, ...]): The current state of the environment.
        q_table (np.ndarray, optional): Array of Q-values for each action. Defaults to None.
        epsilon (float, optional): Probability of choosing a random action. Defaults to 0.1.

    Raises:
        ValueError: If q_table is None.

    Returns:
        int: Index of the selected action.
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_table[(*state,)]))  # Return a random action
    else:  # Return the action with the highest Q-value
        return np.argmax(q_table[(*state,)])
    
    pass

def Q_learning_table_upate(state: Tuple[int, ...] = None, next_state: Tuple[int, ...] = None, action: int = None, reward: float = None, q_table: np.ndarray = None, alpha: float = 0.1, gamma: float = 0.9):
    """
    Updates the Q-table using the Q-learning algorithm.

    Args:
        state (Tuple[int, ...], optional): The current state of the environment. Defaults to None.
        next_state (Tuple[int, ...], optional): The next state of the environment. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        q_table (np.ndarray, optional): Array of Q-values for each state-action pair. Defaults to None.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Raises:
        ValueError: If q_table is None.
        ValueError: If state is None.
        ValueError: If next_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if state is None:
        raise ValueError("state cannot be None!")
    if next_state is None:
        raise ValueError("next_state cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if reward is None:
        raise ValueError("reward cannot be None!")
    
    try:
        q_table[(*state, action)]
    except TypeError as e:
        raise ValueError("state and action must be usable to access the q_table!")
    q_table[(*state, action)] = (q_table[(*state, action)]
    + alpha * (reward 
                + gamma * np.max(q_table[(*next_state, )]) 
                - q_table[(*state, action)]))

    pass