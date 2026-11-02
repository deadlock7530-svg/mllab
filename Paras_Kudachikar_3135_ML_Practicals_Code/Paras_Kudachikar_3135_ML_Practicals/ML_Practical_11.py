# Machine Learning Laboratory - Practical No. 11
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRID_SIZE = 4
GOAL = (3, 3)
OBSTACLE = (1, 1)
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_EFFECTS = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}

Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

def step(state, action_idx):
    dr, dc = ACTION_EFFECTS[ACTIONS[action_idx]]
    r, c = state
    nr = max(0, min(GRID_SIZE-1, r + dr))
    nc = max(0, min(GRID_SIZE-1, c + dc))
    if (nr, nc) == OBSTACLE:
        nr, nc = r, c
    reward = 10 if (nr, nc) == GOAL else -1
    done = (nr, nc) == GOAL
    return (nr, nc), reward, done

rewards_per_episode = []

for ep in range(episodes):
    state = (0, 0)
    total_reward = 0

    for t in range(100):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state, reward, done = step(state, action)
        best_next = np.max(Q[next_state[0], next_state[1]])

        Q[state[0], state[1], action] += alpha * (
            reward + gamma * best_next - Q[state[0], state[1], action]
        )

        state = next_state
        total_reward += reward
        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(0.01, epsilon * 0.995)

print("Training complete over", episodes, "episodes.")
print(f"Average reward (first 50 episodes): {np.mean(rewards_per_episode[:50]):.2f}")
print(f"Average reward (last 50 episodes): {np.mean(rewards_per_episode[-50:]):.2f}")

print("\nLearned Optimal Policy (grid view):")
policy_symbols = {'UP':'^', 'DOWN':'v', 'LEFT':'<', 'RIGHT':'>'}

for r in range(GRID_SIZE):
    row_str = ""
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            row_str += " G "
        elif (r, c) == OBSTACLE:
            row_str += " X "
        else:
            best_a = ACTIONS[np.argmax(Q[r, c])]
            row_str += f" {policy_symbols[best_a]} "
    print(row_str)

window = 20
smoothed = np.convolve(
    rewards_per_episode, np.ones(window) / window, mode='valid'
)

plt.figure(figsize=(7, 4.5))
plt.plot(rewards_per_episode, alpha=0.3, label='Reward per episode')
plt.plot(
    range(window-1, len(rewards_per_episode)), smoothed, linewidth=2, label=f'{window}-episode moving average'
)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Convergence: Grid World (4x4)')
plt.legend()
plt.tight_layout()
plt.savefig('a11_qlearning.png', dpi=130)

print("\nPlot saved. Agent converges to shortest obstacle-avoiding path to goal.")
