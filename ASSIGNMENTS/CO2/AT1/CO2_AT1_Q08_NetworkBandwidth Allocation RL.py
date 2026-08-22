'''Question: Evaluate how Reinforcement Learning can be used in network bandwidth
allocation in telecommunications. Define states, actions, and rewards, and examine
how the system adapts to dynamic conditions.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_USERS = 4
LOAD_BINS = 4
N_STATES = LOAD_BINS ** N_USERS
TOTAL_BANDWIDTH = 100.0
ALLOCATIONS = [
    [40, 20, 20, 20], [25, 25, 25, 25], [50, 20, 15, 15],
    [20, 40, 20, 20], [20, 20, 40, 20], [20, 20, 20, 40]
]
ACTIONS = [f"alloc_{i}" for i in range(len(ALLOCATIONS))]
N_ACTIONS = len(ACTIONS)
def encode_state(loads):
    code = 0
    for l in loads:
        code = code * LOAD_BINS + l
    return code
def simulate_traffic_demand():
    return [random.choices(range(LOAD_BINS), weights=[0.3, 0.3, 0.25, 0.15])[0] for _ in range(N_USERS)]
def compute_bandwidth_reward(action, loads):
    allocation = ALLOCATIONS[action]
    satisfaction, congestion_penalty = 0.0, 0.0
    for i in range(N_USERS):
        demand_mbps = (loads[i] + 1) * 15.0
        if allocation[i] >= demand_mbps:
            satisfaction += 3.0
        else:
            shortfall = demand_mbps - allocation[i]
            congestion_penalty += -0.1 * shortfall
    fairness_penalty = -np.std(allocation) * 0.05
    return satisfaction + congestion_penalty + fairness_penalty
def run_bandwidth_rl(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.12, 0.9, 0.3
    ep_rewards, congestion_log = [], []
    for ep in range(n_episodes):
        loads = simulate_traffic_demand()
        total_reward, congestion_events = 0, 0
        for step in range(15):
            state = encode_state(loads)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            reward = compute_bandwidth_reward(action, loads)
            allocation = ALLOCATIONS[action]
            for i in range(N_USERS):
                if allocation[i] < (loads[i] + 1) * 15.0:
                    congestion_events += 1
            new_loads = simulate_traffic_demand()
            next_state = encode_state(new_loads)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            loads = new_loads
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        congestion_log.append(congestion_events)
    return ep_rewards, congestion_log
print("Bandwidth Allocation RL in Telecommunications")
print(f"State Space: per-user load bin combinations = {N_STATES}")
print(f"Actions (allocation profiles): {ACTIONS}")
rewards, congestion_log = run_bandwidth_rl()
print(f"\nAvg reward first 50 episodes:      {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:       {np.mean(rewards[-50:]):.3f}")
print(f"Avg congestion events first 50:    {np.mean(congestion_log[:50]):.3f}")
print(f"Avg congestion events last 50:     {np.mean(congestion_log[-50:]):.3f}")
print("\nMDP Component Summary:")
print("  States : per-user traffic load level across all connected users")
print("  Actions: discrete bandwidth allocation profiles across users")
print("  Reward : +demand satisfaction - congestion shortfall penalty - fairness imbalance penalty")
print("\nAdaptation to Dynamic Conditions:")
print("  Traffic demand is resampled every step, forcing the agent to generalize across load patterns")
print("  Learned Q-values let the system reallocate bandwidth as usage shifts between users in real time")
print("  Fixed static allocation would leave heavy users congested while light users stay under-utilized")

'''
Output:
Bandwidth Allocation RL in Telecommunications
State Space: per-user load bin combinations = 256
Actions (allocation profiles): ['alloc_0', 'alloc_1', 'alloc_2', 'alloc_3', 'alloc_4', 'alloc_5']

Avg reward first 50 episodes:      -11.988
Avg reward last 50 episodes:       -4.835
Avg congestion events first 50:    37.300
Avg congestion events last 50:     34.500

MDP Component Summary:
  States : per-user traffic load level across all connected users
  Actions: discrete bandwidth allocation profiles across users
  Reward : +demand satisfaction - congestion shortfall penalty - fairness imbalance penalty

Adaptation to Dynamic Conditions:
  Traffic demand is resampled every step, forcing the agent to generalize across load patterns
  Learned Q-values let the system reallocate bandwidth as usage shifts between users in real time
  Fixed static allocation would leave heavy users congested while light users stay under-utilized
'''
