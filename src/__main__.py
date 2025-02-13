"""
__main__.py

Description: This script initializes and runs the main application for the CPE471 Project 1.
Author: Lucas Pinto
Date: February 11, 2025

"""

from __init__ import * # Import everything from the __init__.py file

def main():
    """
    Main function to run the application.
    """

    if __name__ == "__main__":
        print("Hello, World!")
        
        # Environment/Grid World Settings
        grid_length = 100
        grid_width = 100
        reward_vector = [grid_length*grid_width, -1, -5] # In order, the reward for reaching the goal, moving, and an invalid move
        # ^^^ scales dynamically with the grid size
        environment = GridWorld((grid_length, grid_width), None, (grid_length-1, grid_width-1), reward_vector)
        agent_start = (0, 0) # None = random, yet to account for random position in graphing though!

        # Agent Possible Actions
        actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        # Learning Settings
        learning_algorithms = {'Q-Learning': Q_learning_episode, 'Q-Lambda': Q_lambda_episode}
        enable_learning_algorithms = [True, True] # Enable Q-Learning, Q-Lambda, etc...

        # Q-learning Settings
        episodes = 1000
        alpha = 0.15 # Learning rate, how much the agent learns from new information
        gamma = 0.95 # Discount factor, how much the agent values future rewards
        epsilon = 0.1 # Exploration rate, how often the agent explores instead of exploiting

        # Q-Lambda Settings (uses ^^^ settings)
        lambda_value = 0.5 # Lambda value for Q-Lambda learning

        # Enable recording of action sequence, total rewards, steps taken, and Q-table history
        enable_record_set_1 = [True, True, True, True] # Applies to first and last episode
        enable_record_set_2 = [False, True, True, False] # Applies to everything between first and last episode
        
        # Plotting Settings
        fps = 0 # Frames per second for the plot animation, disables animation at 0

        enable_q_table_plots = False # Enable Q-table plots
        enable_episode_plots = False # Enable episode plots such as rewards/steps over time
        enable_first_action_sequence_plots = True
        enable_last_action_sequence_plots = True

        training_settings_summary = f"{grid_length}x{grid_width} Grid World\nEpisodes: {episodes}, Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\nRewards: {reward_vector}"
        agent_settings_summary = f"Agent Start: (0, 0), Goal: ({grid_length-1}, {grid_width-1})"
        algorithm_settings_summary = None

        for algorithm_name, algorithm_function in learning_algorithms.items():

            algorithm_settings_summary = f"Trained w/ {algorithm_name} and Epsilon-Greedy Selection"

            # Initialize Q-table with zeros
            q_table = np.zeros((grid_length, grid_width, len(actions)), dtype = float) # Initialize Q-table with zeros

            training_data = []
            
            enable_record = enable_record_set_1
            
            if enable_learning_algorithms[list(learning_algorithms.keys()).index(algorithm_name)]:
                for episode in range(episodes):
                    environment.reset()
                    if (episode == 0) or (episode == episodes - 1):
                        enable_record = enable_record_set_1
                    else:
                        enable_record = enable_record_set_2

                    print(f"Training {algorithm_name} agent Episode {episode + 1} of {episodes}...", end=' ')
                    # Run a single episode of the learning algorithm
                    action_sequence, total_reward, steps_taken, q_table_history = None, None, None, None
                    if algorithm_name == 'Q-Learning':
                        action_sequence, total_reward, steps_taken, q_table_history = Q_learning_episode(
                            environment, None, actions, q_table, 
                            epsilon_greedy_selection, {'q_table': q_table, 'epsilon': epsilon},
                            alpha, gamma, agent_start, enable_record)
                    elif algorithm_name == 'Q-Lambda':
                        action_sequence, total_reward, steps_taken, q_table_history = Q_lambda_episode(
                            environment, None, actions, q_table, 
                            epsilon_greedy_selection, {'q_table': q_table, 'epsilon': epsilon},
                            alpha, gamma, lambda_value, agent_start, enable_record)
                    
                    training_data.append([action_sequence, total_reward, steps_taken, q_table_history])
                    print(f"Completed!!! Total Reward: {total_reward}, Steps Taken: {steps_taken}.")

                print(f"{algorithm_name} Training completed.")

                # Extract total rewards and steps taken per episode
                total_rewards = [data[1] for data in training_data]
                steps_taken = [data[2] for data in training_data]

                # Extract the first and last Q-tables
                first_q_table = training_data[0][3]
                last_q_table = training_data[-1][3]

                if(grid_length*grid_width <= 25) and (enable_q_table_plots): # Too high and the q_Table simply crashes the program
                    plot_q_table(first_q_table, grid_length, grid_width, 
                                actions, 'First Q-table', 
                                training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + algorithm_settings_summary)
                    
                    plot_q_table(last_q_table, grid_length, grid_width, 
                                actions, 'Last Q-table', 
                                training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + algorithm_settings_summary)
                else: 
                    print("Grid too large to display Q-tables. Try to keep the area under 25 cells.")

                if(enable_episode_plots):
                    # Plot total rewards per episode
                    plot_episode_data(total_rewards, episodes, 'Total Reward per Episode', 
                                    training_settings_summary
                                        + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary,
                                            ylabel='Total Reward', label='Total Reward', color='blue')

                    # Plot steps taken per episode
                    plot_episode_data(steps_taken, episodes, 'Steps Taken per Episode',
                                    training_settings_summary
                                        + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary,
                                            ylabel='Steps Taken', label='Steps Taken', color='orange')

                if(enable_first_action_sequence_plots):
                    # Plot the first action sequence
                    first_action_sequence = training_data[0][0]
                    plot_action_sequence(first_action_sequence, grid_length, grid_width, 
                                        'First Action Sequence', 
                                        (training_settings_summary
                                        + "\n" + agent_settings_summary
                                            + "\n" + algorithm_settings_summary),
                                            fps=fps)

                if(enable_last_action_sequence_plots):
                    # Plot the last action sequence
                    last_action_sequence = training_data[-1][0]
                    plot_action_sequence(last_action_sequence, grid_length, grid_width, 
                                        'Last Action Sequence', 
                                        (training_settings_summary
                                        + "\n" + agent_settings_summary
                                            + "\n" + algorithm_settings_summary),
                                            fps=fps)
            
    pass

main()