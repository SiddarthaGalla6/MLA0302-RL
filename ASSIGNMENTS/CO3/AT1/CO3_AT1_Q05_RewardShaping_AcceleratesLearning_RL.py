'''Question: Explain how reward shaping can accelerate learning in RL and illustrate its impact
on preventing suboptimal policies.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
GRID_SIZE = 6
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
GOAL = (5, 5)
def state_id(r, c):
    return r * GRID_SIZE + c
def sparse_reward(r, c):
    return 20.0 if (r, c) == GOAL else -0.1
def shaped_reward(r, c, prev_r, prev_c):
    if (r, c) == GOAL:
        return 20.0
    prev_dist = abs(prev_r - GOAL[0]) + abs(prev_c - GOAL[1])
    new_dist = abs(r - GOAL[0]) + abs(c - GOAL[1])
    shaping_bonus = (prev_dist - new_dist) * 0.5
    return -0.1 + shaping_bonus
def run_gridworld_rl(reward_mode, n_episodes=300):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    episodes_to_solve, ep_rewards = [], []
    for ep in range(n_episodes):
        r, c = 0, 0
        total_reward, steps = 0, 0
        for step in range(60):
            state = state_id(r, c)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            dr, dc = moves[action]
            nr, nc = min(max(r + dr, 0), GRID_SIZE - 1), min(max(c + dc, 0), GRID_SIZE - 1)
            if reward_mode == 'sparse':
                reward = sparse_reward(nr, nc)
            else:
                reward = shaped_reward(nr, nc, r, c)
            next_state = state_id(nr, nc)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            r, c = nr, nc
            total_reward += reward
            steps += 1
            if (r, c) == GOAL:
                break
        epsilon = max(0.05, epsilon * 0.99)
        episodes_to_solve.append(steps)
        ep_rewards.append(total_reward)
    return episodes_to_solve, ep_rewards
print("Reward Shaping Impact on RL Learning Speed")
print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}, Goal: {GOAL}")
sparse_steps, sparse_rewards = run_gridworld_rl('sparse')
shaped_steps, shaped_rewards = run_gridworld_rl('shaped')
print(f"\n{'RewardType':>12} {'AvgStepsFirst30':>17} {'AvgStepsLast30':>16} {'AvgRewardLast30':>17}")
print("-" * 68)
print(f"{'Sparse':>12} {np.mean(sparse_steps[:30]):>17.2f} {np.mean(sparse_steps[-30:]):>16.2f} {np.mean(sparse_rewards[-30:]):>17.2f}")
print(f"{'Shaped':>12} {np.mean(shaped_steps[:30]):>17.2f} {np.mean(shaped_steps[-30:]):>16.2f} {np.mean(shaped_rewards[-30:]):>17.2f}")
print("\nReward Shaping Explanation:")
print("  Sparse reward only signals success at the goal, leaving early episodes almost purely random exploration")
print("  Shaped reward adds a distance-based bonus that guides the agent toward the goal at every step")
print("  This denser signal accelerates convergence, reducing average steps-to-goal much faster than sparse reward")
print("\nPreventing Suboptimal Policies:")
print("  Careless shaping (e.g. rewarding raw distance reduction without care) can create loops chasing bonus, not the goal")
print("  Potential-based shaping (used here as difference in distance) preserves the optimal policy while speeding up learning")

'''
Output:
Reward Shaping Impact on RL Learning Speed
Grid size: 6x6, Goal: (5, 5)

  RewardType   AvgStepsFirst30   AvgStepsLast30   AvgRewardLast30
--------------------------------------------------------------------
      Sparse             40.93            10.33             19.07
      Shaped             16.43            10.53             23.55

Reward Shaping Explanation:
  Sparse reward only signals success at the goal, leaving early episodes almost purely random exploration
  Shaped reward adds a distance-based bonus that guides the agent toward the goal at every step
  This denser signal accelerates convergence, reducing average steps-to-goal much faster than sparse reward

Preventing Suboptimal Policies:
  Careless shaping (e.g. rewarding raw distance reduction without care) can create loops chasing bonus, not the goal
  Potential-based shaping (used here as difference in distance) preserves the optimal policy while speeding up learning
'''
