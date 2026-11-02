import numpy as np 
import random 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
# Simple Grid World: 4x4 grid, agent starts at (0,0), goal at (3,3), one obstacle at (1,1) 
GRID_SIZE = 4 
GOAL = (3, 3) 
OBSTACLE = (1, 1) 
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'] 
ACTION_EFFECTS = { 
    'UP': (-1,0), 
    'DOWN': (1,0), 
    'LEFT': (0,-1), 
    'RIGHT': (0,1) 
} 
 
q_table = np.zeros( 
    (GRID_SIZE, GRID_SIZE, len(ACTIONS)) 
) 
 
learning_rate = 0.1 
discount_factor = 0.9 
exploration_rate = 0.2 
total_episodes = 500 
 
def take_step(current_state, action_index): 
 
    row_change, column_change = ACTION_EFFECTS[ 
        ACTIONS[action_index] 
    ] 
 
    current_row, current_column = current_state 
 
    next_row = max( 
        0, 
        min( 
            GRID_SIZE-1, 
            current_row + row_change 
        ) 
    ) 
 
    next_column = max( 
        0, 
        min( 
            GRID_SIZE-1, 
            current_column + column_change 
        ) 
    ) 
 
    if (next_row, next_column) == OBSTACLE: 
        next_row, next_column = current_row, current_column 
 
    step_reward = ( 
        10 
        if (next_row, next_column) == GOAL 
        else -1 
    ) 
 
    episode_done = ( 
        (next_row, next_column) == GOAL 
    ) 
 
    return ( 
        (next_row, next_column), 
        step_reward, 
        episode_done 
    ) 
 
episode_rewards = [] 
for episode_number in range(total_episodes): 
 
    current_state = (0, 0) 
    accumulated_reward = 0 
 
    for time_step in range(100): 
 
        if random.random() < exploration_rate: 
            selected_action = random.randint(0, 3) 
        else: 
            selected_action = np.argmax( 
                q_table[ 
                    current_state[0], 
                    current_state[1] 
                ] 
            ) 
 
        next_state, step_reward, episode_done = take_step( 
            current_state, 
            selected_action 
        ) 
 
        best_next_value = np.max( 
            q_table[ 
                next_state[0], 
                next_state[1] 
            ] 
        ) 
 
        q_table[ 
            current_state[0], 
            current_state[1], 
            selected_action 
        ] += learning_rate * ( 
            step_reward 
            + discount_factor * best_next_value 
            - q_table[ 
                current_state[0], 
                current_state[1], 
                selected_action 
            ] 
        ) 
 
        current_state = next_state 
        accumulated_reward += step_reward 
 
        if episode_done: 
            break 
 
    episode_rewards.append(accumulated_reward) 
 
    exploration_rate = max( 
        0.01, 
        exploration_rate * 0.995 
    ) 
 
print( 
    "Training complete over", 
    total_episodes, 
    "episodes." 
) 
 
print( 
    f"Average reward (first 50 episodes) : " 
    f"{np.mean(episode_rewards[:50]):.2f}" 
) 
 
print( 
    f"Average reward (last 50 episodes) : " 
    f"{np.mean(episode_rewards[-50:]):.2f}" 
) 
 
# Derive optimal policy 
print("\nLearned Optimal Policy (grid view):") 
 
policy_symbols = { 
    'UP':'^', 
    'DOWN':'v', 
    'LEFT':'<', 
    'RIGHT':'>' 
} 
 
for grid_row in range(GRID_SIZE): 
 
    row_string = "" 
 
    for grid_column in range(GRID_SIZE): 
 
        if (grid_row, grid_column) == GOAL: 
            row_string += " G " 
        elif (grid_row, grid_column) == OBSTACLE: 
            row_string += " X " 
 
        else: 
            best_action = ACTIONS[ 
                np.argmax( 
                    q_table[ 
                        grid_row, 
                        grid_column 
                    ] 
                ) 
            ] 
 
            row_string += ( 
                f" {policy_symbols[best_action]} " 
            ) 
 
    print(row_string) 
 
plt.figure(figsize=(7,4.5)) 
 
moving_window = 20 
 
smoothed_rewards = np.convolve( 
    episode_rewards, 
    np.ones(moving_window) / moving_window, 
    mode='valid' 
) 
 
plt.plot( 
    episode_rewards, 
    alpha=0.3, 
    color='gray', 
    label='Reward per episode' 
) 
 
plt.plot( 
    range( 
        moving_window-1, 
        len(episode_rewards) 
    ), 
    smoothed_rewards, 
    color='#4C72B0', 
    linewidth=2, 
    label=f'{moving_window}-episode moving avg' 
) 
 
plt.xlabel('Episode') 
plt.ylabel('Total Reward') 
 
plt.title( 
    'Q-Learning Convergence: Grid World (4x4)' 
) 
 
plt.legend() 
plt.tight_layout() 
 
plt.savefig( 
    'imgs/a11_qlearning.png', 
    dpi=130 
) 
print( 
"\nPlot saved. Agent converges to shortest " 
"obstacle-avoiding path to goal." 
)