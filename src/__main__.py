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

    print("Hello, World!")

    if __name__ == "__main__":
        print()
        grid_length = 100
        grid_width = 100

        episodes = 100
        alpha = 0.15
        gamma = 0.95
        epsilon = 0

        reward_vector = [10, -0.1, -1] # In order, the reward for reaching the goal, moving, and an invalid move
        actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        environment = GridWorld((grid_length, grid_width), None, (grid_length-1, grid_width-1), reward_vector)
        q_table = np.zeros((grid_length, grid_width, len(actions)), dtype = float) # Initialize Q-table with zeros
        
        training_data = []

        print("Training Q-Learning agent...")
        
        for episode in range(episodes):
            print(f"Episode {episode+1}/{episodes}")
            training_data.append(Q_learning_episode(grid_world=environment, 
                            agent=None, 
                            actions=actions,
                            q_table=q_table,
                            selection_function=epsilon_greedy_selection,
                            function_args={'q_table': q_table, 'epsilon': epsilon},
                            alpha=alpha, 
                            gamma=gamma, 
                            agent_start=(0, 0)))
        print("Q-Learning Training completed.")

        # Extract total rewards and steps taken per episode
        total_rewards = [data[1] for data in training_data]
        steps_taken = [data[2] for data in training_data]

        # Plot total rewards per episode
        plt.figure(figsize=(12, 6))
        plt.plot(range(episodes), total_rewards, label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.show()

        # Plot steps taken per episode
        plt.figure(figsize=(12, 6))
        plt.plot(range(episodes), steps_taken, label='Steps Taken', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Steps Taken')
        plt.title('Steps Taken per Episode')
        plt.legend()
        plt.show()
    pass

main()