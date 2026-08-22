'''Question: Illustrate the use of Reinforcement Learning in smart city waste management
systems. Identify the MDP components and discuss how continuous updates improve
operational efficiency.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_BINS = 5
FILL_BINS = 4
N_STATES = FILL_BINS ** N_BINS
ACTIONS = [f"collect_bin_{i}" for i in range(N_BINS)] + ['skip_round']
N_ACTIONS = len(ACTIONS)
def encode_fill_levels(fill_levels):
    code = 0
    for f in fill_levels:
        code = code * FILL_BINS + f
    return code
def simulate_fill_rate(bin_id):
    rates = [0.3, 0.5, 0.2, 0.6, 0.4]
    return rates[bin_id]
def compute_waste_reward(action, fill_levels, distance_cost):
    if action == N_ACTIONS - 1:
        overflow_penalty = sum(-8.0 for f in fill_levels if f == FILL_BINS - 1)
        return overflow_penalty
    bin_id = action
    fill = fill_levels[bin_id]
    if fill == FILL_BINS - 1:
        collection_value = 10.0
    elif fill >= FILL_BINS - 2:
        collection_value = 4.0
    else:
        collection_value = -2.0
    return collection_value - distance_cost
def run_waste_rl(n_episodes=400):
    Q = {}
    def get_q(s, a):
        return Q.get((s, a), 0.0)
    def set_q(s, a, v):
        Q[(s, a)] = v
    alpha, gamma, epsilon = 0.15, 0.9, 0.3
    ep_rewards, overflow_events = [], []
    for ep in range(n_episodes):
        fill_levels = [random.randint(0, 1) for _ in range(N_BINS)]
        total_reward, overflow_count = 0, 0
        for step in range(15):
            state = encode_fill_levels(fill_levels)
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                q_vals = [get_q(state, a) for a in range(N_ACTIONS)]
                action = int(np.argmax(q_vals))
            distance_cost = 1.5 if action < N_BINS else 0.0
            reward = compute_waste_reward(action, fill_levels, distance_cost)
            new_fill_levels = list(fill_levels)
            if action < N_BINS:
                new_fill_levels[action] = 0
            for i in range(N_BINS):
                if random.random() < simulate_fill_rate(i):
                    new_fill_levels[i] = min(new_fill_levels[i] + 1, FILL_BINS - 1)
            overflow_count += sum(1 for f in new_fill_levels if f == FILL_BINS - 1)
            next_state = encode_fill_levels(new_fill_levels)
            next_q_vals = [get_q(next_state, a) for a in range(N_ACTIONS)]
            old_q = get_q(state, action)
            set_q(state, action, old_q + alpha * (reward + gamma * max(next_q_vals) - old_q))
            fill_levels = new_fill_levels
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        overflow_events.append(overflow_count)
    return ep_rewards, overflow_events
print("Smart City Waste Management using Reinforcement Learning")
print(f"State Space: fill level combinations across {N_BINS} bins, up to {N_STATES} states")
print(f"Actions: {ACTIONS}")
rewards, overflow_events = run_waste_rl()
print(f"\nAvg reward first 50 episodes:    {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:     {np.mean(rewards[-50:]):.3f}")
print(f"Avg overflow events first 50:    {np.mean(overflow_events[:50]):.3f}")
print(f"Avg overflow events last 50:     {np.mean(overflow_events[-50:]):.3f}")
print("\nMDP Component Summary:")
print("  States : fill level of every bin in the network")
print("  Actions:", ACTIONS)
print("  Reward : +overflow-prevention value - travel distance cost - overflow penalty")
print("\nContinuous Update Benefit:")
print("  Real-time sensor fill readings each round let the agent prioritize near-full bins")
print("  Continuous retraining reduces overflow events over time, cutting missed collections")
print("  Static fixed-route collection cannot adapt to uneven fill rates the way this policy does")

'''
Output:
Smart City Waste Management using Reinforcement Learning
State Space: fill level combinations across 5 bins, up to 1024 states
Actions: ['collect_bin_0', 'collect_bin_1', 'collect_bin_2', 'collect_bin_3', 'collect_bin_4', 'skip_round']

Avg reward first 50 episodes:    -7.690
Avg reward last 50 episodes:     39.730
Avg overflow events first 50:    18.480
Avg overflow events last 50:     6.560

MDP Component Summary:
  States : fill level of every bin in the network
  Actions: ['collect_bin_0', 'collect_bin_1', 'collect_bin_2', 'collect_bin_3', 'collect_bin_4', 'skip_round']
  Reward : +overflow-prevention value - travel distance cost - overflow penalty

Continuous Update Benefit:
  Real-time sensor fill readings each round let the agent prioritize near-full bins
  Continuous retraining reduces overflow events over time, cutting missed collections
  Static fixed-route collection cannot adapt to uneven fill rates the way this policy does
'''
