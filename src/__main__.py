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
        reward_vector = [grid_length*grid_width, -1, -(grid_length*grid_width)/10.0] # In order, the reward for reaching the goal, moving, and an invalid move
        # ^^^ scales dynamically with the grid size
        environment = GridWorld((grid_length, grid_width), None, (grid_length-1, grid_width-1), reward_vector)

        # Agent Possible Actions
        actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        # Q-learning Settings
        episodes = 10000
        alpha = 0.33 # Learning rate, how much the agent learns from new information
        gamma = 0.80 # Discount factor, how much the agent values future rewards
        epsilon = 0.1 # Exploration rate, how often the agent explores instead of exploiting

        # Enable recording of action sequence, total rewards, steps taken, and Q-table history
        enable_record_set_1 = [True, True, True, True] # Applies to first and last episode
        enable_record_set_2 = [False, True, True, False] # Applies to everything between first and last episode
        
        # Plotting Settings
        fps = 6000 # Frames per second for the plot animation, disables animation at 0

        enable_q_table_plots = False # Enable Q-table plots
        enable_episode_plots = False # Enable episode plots such as rewards/steps over time
        enable_first_action_sequence_plots = False
        enable_last_action_sequence_plots = True

        training_settings_summary = f"{grid_length}x{grid_width} Grid World\nEpisodes: {episodes}, Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\nRewards: {reward_vector}"
        agent_settings_summary = f"Agent Start: (0, 0), Goal: ({grid_length-1}, {grid_width-1})"
        manual_settings_summary = "Trained w/ Q-Leaning and Epsilon-Greedy Selection"

        # Initialize Q-table with zeros
        q_table = np.zeros((grid_length, grid_width, len(actions)), dtype = float) # Initialize Q-table with zeros
        
        training_data = []

        print("Training Q-Learning agent...")
        
        enable_record = enable_record_set_1

        for episode in range(episodes):
            if(episode == 0) or (episode == episodes-1):
                enable_record = enable_record_set_1
            else:
                enable_record = enable_record_set_2

            print(f"Training Q-Learning agent Episode {episode+1} of {episodes}...", end=' ')
            # Run a single episode of Q-learning
            # Per episode returns a list of, in order: action sequence, total reward, steps taken, and Q-table history.
            action_sequence, total_rewards, steps_taken, q_table_history = Q_learning_episode(grid_world=environment, 
                                                                                                agent=None, 
                                                                                                actions=actions,
                                                                                                q_table=q_table,
                                                                                                selection_function=epsilon_greedy_selection,
                                                                                                function_args={'q_table': q_table, 'epsilon': epsilon},
                                                                                                alpha=alpha, 
                                                                                                gamma=gamma, 
                                                                                                agent_start=(0, 0), 
                                                                                                enable_record=enable_record)
            training_data.append([action_sequence, total_rewards, steps_taken, q_table_history])
            print(f"Completed!!! Total Reward: {total_rewards}, Steps Taken: {steps_taken}.")
            
        print("Q-Learning Training completed.")

        # Extract total rewards and steps taken per episode
        total_rewards = [data[1] for data in training_data]
        steps_taken = [data[2] for data in training_data]

        # Extract the first and last Q-tables
        first_q_table = training_data[0][3]
        last_q_table = training_data[-1][3]

        # Convert Q-tables to 2D arrays for visualization
        first_q_table_2d = q_table_to_2d_array(first_q_table, grid_length, grid_width)
        last_q_table_2d = q_table_to_2d_array(last_q_table, grid_length, grid_width)

        if(grid_length*grid_width <= 25) and (enable_q_table_plots): # Too high and the q_Table simply crashes the program

            # Plot the first Q-table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            table = ax.table(cellText=first_q_table_2d, colLabels=["State(x,y)"] + list(actions.keys()), loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title('First Q-table')
            plt.suptitle(training_settings_summary + "\n" + agent_settings_summary + "\n" + manual_settings_summary, fontsize=8)
            plt.show()

            # Plot the last Q-table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            table = ax.table(cellText=last_q_table_2d, colLabels=["State(x,y)"] + list(actions.keys()), loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title('Last Q-table')
            plt.suptitle(training_settings_summary + "\n" + agent_settings_summary + "\n" + manual_settings_summary, fontsize=8)
            plt.show()
        
        else: 
            print("Grid too large to display Q-tables. Try to keep the area under 25 cells.")

        if(enable_episode_plots):
            # Plot total rewards per episode
            plt.figure(figsize=(12, 10))
            plt.plot(range(episodes), total_rewards, label='Total Reward')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward per Episode')
            plt.suptitle(training_settings_summary + "\n" + agent_settings_summary + "\n" + manual_settings_summary, fontsize=8)
            plt.legend()
            plt.show()

            # Plot steps taken per episode
            plt.figure(figsize=(12, 8))
            plt.plot(range(episodes), steps_taken, label='Steps Taken', color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Steps Taken')
            plt.title('Steps Taken per Episode')
            plt.suptitle(training_settings_summary + "\n" + agent_settings_summary + "\n" + manual_settings_summary, fontsize=8)
            plt.legend()
            plt.show()

        if(enable_first_action_sequence_plots):
            # Plot the first action sequence
            first_action_sequence = training_data[0][0]
            plot_action_sequence(first_action_sequence, 
                                grid_length, grid_width, 
                                'First Action Sequence', 
                                (training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + manual_settings_summary),
                                    fps=fps)

        if(enable_last_action_sequence_plots):
            # Plot the last action sequence
            last_action_sequence = training_data[-1][0]
            plot_action_sequence(last_action_sequence, 
                                grid_length, grid_width, 
                                'Last Action Sequence', 
                                (training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + manual_settings_summary),
                                    fps=fps)
    pass

main()