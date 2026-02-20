#q learning agent

import numpy as np

grid_size = (5, 5)
num_states = grid_size[0] * grid_size[1]

actions = ["up", "down", "left", "right"]
num_actions = len(actions)

def state_to_coordinates(state):
    return divmod(state, grid_size[1])

def coordinates_to_state(x, y):
    return x * grid_size[1] + y

rewards = np.full(num_states, -1)
terminal_state = coordinates_to_state(4, 4)
rewards[terminal_state] = 100

def take_action(state, action):
    x, y = state_to_coordinates(state)

    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(grid_size[0] - 1, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(grid_size[1] - 1, y + 1)

    return coordinates_to_state(x, y)

q_table = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 500

for episode in range(num_episodes):
    state = np.random.randint(0, num_states)

    while state != terminal_state:
        if np.random.rand() < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(q_table[state])

        next_state = take_action(state, actions[action])
        reward = rewards[next_state]

        best_next_action = np.max(q_table[next_state])

        q_table[state, action] += alpha * (
            reward + gamma * best_next_action - q_table[state, action]
        )

        state = next_state

policy = np.argmax(q_table, axis=1)
policy_grid = np.array([actions[a] for a in policy]).reshape(grid_size)

print("Optimal Policy:")
print(policy_grid)

print("\nQ-Table:")
print(q_table)