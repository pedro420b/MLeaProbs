import numpy as np
import random

# Define the environment
grid = [['S', ' ', ' ', ' '],
        [' ', 'X', ' ', 'X'],
        [' ', ' ', ' ', ' '],
        [' ', 'X', 'X', 'G']]

rows = len(grid)
cols = len(grid[0])
actions = ['up', 'down', 'left', 'right']

# Q-Learning parameters
alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

# Initialize Q-table
q_table = np.zeros((rows * cols, len(actions)))

def get_state(row, col):
    return row * cols + col

def get_next_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state, :])]

def get_next_location(row, col, action):
    new_row, new_col = row, col
    if action == 'up' and row > 0:
        new_row -= 1
    elif action == 'down' and row < rows - 1:
        new_row += 1
    elif action == 'left' and col > 0:
        new_col -= 1
    elif action == 'right' and col < cols - 1:
        new_col += 1
    return new_row, new_col

def get_reward(row, col):
    if grid[row][col] == 'G':
        return 10
    elif grid[row][col] == 'X':
        return -10
    else:
        return -1

# Training loop
for episode in range(episodes):
    row, col = 0, 0  # Start at the top-left corner
    while grid[row][col] != 'G':
        state = get_state(row, col)
        action = get_next_action(state)
        next_row, next_col = get_next_location(row, col, action)
        reward = get_reward(next_row, next_col)
        next_state = get_state(next_row, next_col)

        # Update Q-table
        old_value = q_table[state, actions.index(action)]
        next_max = np.max(q_table[next_state, :])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, actions.index(action)] = new_value

        row, col = next_row, next_col

print("Training finished.\n")

# Output the learned Q-table (Optional)
print(q_table)

# Use the trained Q-table for navigation
row, col = 0, 0  # Start at the top-left corner
steps = 0
while grid[row][col] != 'G':
    state = get_state(row, col)
    action = actions[np.argmax(q_table[state, :])]  # Exploit the learned policy (no exploration)
    next_row, next_col = get_next_location(row, col, action)

    print(f"Step {steps}: From ({row}, {col}) to ({next_row}, {next_col}) taking action {action}")
    
    row, col = next_row, next_col
    steps += 1

print(f"\nReached the goal in {steps} steps!")
