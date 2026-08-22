'''Question: Explain how Reinforcement Learning can be applied to optimize delivery routes in
a logistics system. Identify the possible states, actions, rewards, and policy. Discuss
how continuous learning improves efficiency over time.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_NODES = 6
N_STATES = N_NODES * N_NODES
ACTIONS = ['go_node_0', 'go_node_1', 'go_node_2', 'go_node_3', 'go_node_4', 'go_node_5']
N_ACTIONS = len(ACTIONS)
DEPOT = 0
distance_matrix = np.array([
    [0, 4, 8, 6, 10, 5],
    [4, 0, 5, 9, 7, 3],
    [8, 5, 0, 4, 6, 8],
    [6, 9, 4, 0, 3, 7],
    [10, 7, 6, 3, 0, 4],
    [5, 3, 8, 7, 4, 0]
])
traffic_multiplier = np.ones((N_NODES, N_NODES))
def encode_state(current_node, visited_mask):
    return current_node * (2 ** N_NODES) + visited_mask
Q = {}
def get_q(state, action):
    return Q.get((state, action), 0.0)
def set_q(state, action, value):
    Q[(state, action)] = value
def compute_delivery_reward(current_node, next_node, visited_mask, traffic):
    if (visited_mask >> next_node) & 1:
        return -20.0
    dist = distance_matrix[current_node, next_node] * traffic[current_node, next_node]
    time_penalty = -dist
    delivery_bonus = 10.0
    fuel_penalty = -0.5 * dist
    return time_penalty + delivery_bonus + fuel_penalty
def simulate_traffic():
    return traffic_multiplier * (0.8 + np.random.rand(N_NODES, N_NODES) * 0.6)
def run_delivery_rl(n_episodes=400):
    alpha, gamma, epsilon = 0.1, 0.9, 0.6
    ep_rewards, ep_distances = [], []
    for ep in range(n_episodes):
        traffic = simulate_traffic()
        current_node = DEPOT
        visited_mask = 1 << DEPOT
        total_r, total_dist = 0, 0
        for step in range(N_NODES):
            state = encode_state(current_node, visited_mask)
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                q_vals = [get_q(state, a) for a in range(N_ACTIONS)]
                action = int(np.argmax(q_vals))
            reward = compute_delivery_reward(current_node, action, visited_mask, traffic)
            next_visited = visited_mask | (1 << action)
            next_state = encode_state(action, next_visited)
            next_q_vals = [get_q(next_state, a) for a in range(N_ACTIONS)]
            old_q = get_q(state, action)
            new_q = old_q + alpha * (reward + gamma * max(next_q_vals) - old_q)
            set_q(state, action, new_q)
            if not ((visited_mask >> action) & 1):
                total_dist += distance_matrix[current_node, action]
                current_node = action
                visited_mask = next_visited
            total_r += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_r)
        ep_distances.append(total_dist)
    return ep_rewards, ep_distances
print("Delivery Route Optimization using Reinforcement Learning")
print(f"State Space: current_node x visited_subset, N_STATES up to {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, distances = run_delivery_rl()
print(f"{'Phase':>15} {'Avg Reward':>12} {'Avg Distance':>14}")
print("-" * 45)
print(f"{'Early (0-50)':>15} {np.mean(rewards[:50]):>12.3f} {np.mean(distances[:50]):>14.3f}")
print(f"{'Mid (150-200)':>15} {np.mean(rewards[150:200]):>12.3f} {np.mean(distances[150:200]):>14.3f}")
print(f"{'Late (350-400)':>15} {np.mean(rewards[350:400]):>12.3f} {np.mean(distances[350:400]):>14.3f}")
print("\nMDP Component Summary:")
print("  States : current node combined with set of already-visited nodes")
print("  Actions: move to any of the delivery nodes")
print("  Reward : -travel_time - fuel_cost + delivery_bonus - revisit_penalty")
print("  Policy : greedy selection over learned Q-values after training")
print("\nContinuous Learning Benefit:")
reward_gain = np.mean(rewards[350:400]) - np.mean(rewards[:50])
print(f"  Average episode reward improved by {reward_gain:.2f} from early to late training")
print("  Live traffic sampled each episode lets the agent adapt routes to changing conditions")

'''
Output:
Delivery Route Optimization using Reinforcement Learning
State Space: current_node x visited_subset, N_STATES up to 36
Actions: ['go_node_0', 'go_node_1', 'go_node_2', 'go_node_3', 'go_node_4', 'go_node_5']
          Phase   Avg Reward   Avg Distance
---------------------------------------------
   Early (0-50)      -35.664         17.960
  Mid (150-200)       -4.958         19.760
 Late (350-400)       -7.641         20.880

MDP Component Summary:
  States : current node combined with set of already-visited nodes
  Actions: move to any of the delivery nodes
  Reward : -travel_time - fuel_cost + delivery_bonus - revisit_penalty
  Policy : greedy selection over learned Q-values after training

Continuous Learning Benefit:
  Average episode reward improved by 28.02 from early to late training
  Live traffic sampled each episode lets the agent adapt routes to changing conditions
'''
