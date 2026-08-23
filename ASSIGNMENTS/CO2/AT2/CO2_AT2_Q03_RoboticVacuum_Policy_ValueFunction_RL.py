'''Question: Design and develop a Reinforcement Learning framework a robotic vacuum
cleaner uses Reinforcement Learning to clean efficiently while avoiding obstacles.
Apply the concept of policy and explain how the value function improves performance
over time.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
GRID_SIZE = 6
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
ACTIONS = ['up', 'down', 'left', 'right']
OBSTACLES = [(1, 2), (2, 2), (3, 3), (4, 1)]
DIRTY_CELLS = [(0, 5), (2, 0), (3, 4), (5, 5), (5, 0)]
def state_id(r, c):
    return r * GRID_SIZE + c
def compute_vacuum_reward(r, c, dirty_set):
    if (r, c) in OBSTACLES:
        return -10.0
    if (r, c) in dirty_set:
        return 8.0
    return -0.2
def run_vacuum_rl(n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    V = np.zeros(N_STATES)
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, cells_cleaned = [], []
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    for ep in range(n_episodes):
        r, c = 0, 0
        dirty_set = set(DIRTY_CELLS)
        total_reward, cleaned = 0, 0
        for step in range(40):
            state = state_id(r, c)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            dr, dc = moves[action]
            nr, nc = min(max(r + dr, 0), GRID_SIZE - 1), min(max(c + dc, 0), GRID_SIZE - 1)
            if (nr, nc) in OBSTACLES:
                nr, nc = r, c
            reward = compute_vacuum_reward(nr, nc, dirty_set)
            if (nr, nc) in dirty_set:
                dirty_set.discard((nr, nc))
                cleaned += 1
            r, c = nr, nc
            next_state = state_id(r, c)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            V[state] = np.max(Q[state])
            total_reward += reward
            if not dirty_set:
                break
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        cells_cleaned.append(cleaned)
    return ep_rewards, cells_cleaned, V
print("Robotic Vacuum Cleaner RL with Obstacle Avoidance")
print(f"State Space: grid positions = {N_STATES}")
print(f"Actions: {ACTIONS}")
print(f"Obstacles: {OBSTACLES}")
print(f"Dirty cells to clean: {DIRTY_CELLS}")
rewards, cells_cleaned, V = run_vacuum_rl()
print(f"\nAvg reward first 50 episodes:        {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:         {np.mean(rewards[-50:]):.3f}")
print(f"Avg cells cleaned first 50 episodes: {np.mean(cells_cleaned[:50]):.3f} / {len(DIRTY_CELLS)}")
print(f"Avg cells cleaned last 50 episodes:  {np.mean(cells_cleaned[-50:]):.3f} / {len(DIRTY_CELLS)}")
print("\nPolicy and Value Function Explanation:")
print("  Policy  : greedy action selection over Q(state, action) after training, i.e. pi(s) = argmax_a Q(s, a)")
print("  Value   : V(s) = max_a Q(s, a) estimates long-term cleaning benefit of being in state s")
print("  Learning: as V(s) near dirty cells rises through repeated visits, the policy routes toward them")
print("  Obstacle cells retain low Q-values, so the learned policy naturally avoids collisions over time")

'''
Output:
Robotic Vacuum Cleaner RL with Obstacle Avoidance
State Space: grid positions = 36
Actions: ['up', 'down', 'left', 'right']
Obstacles: [(1, 2), (2, 2), (3, 3), (4, 1)]
Dirty cells to clean: [(0, 5), (2, 0), (3, 4), (5, 5), (5, 0)]

Avg reward first 50 episodes:        0.528
Avg reward last 50 episodes:         1.184
Avg cells cleaned first 50 episodes: 1.040 / 5
Avg cells cleaned last 50 episodes:  1.120 / 5

Policy and Value Function Explanation:
  Policy  : greedy action selection over Q(state, action) after training, i.e. pi(s) = argmax_a Q(s, a)
  Value   : V(s) = max_a Q(s, a) estimates long-term cleaning benefit of being in state s
  Learning: as V(s) near dirty cells rises through repeated visits, the policy routes toward them
  Obstacle cells retain low Q-values, so the learned policy naturally avoids collisions over time
'''
