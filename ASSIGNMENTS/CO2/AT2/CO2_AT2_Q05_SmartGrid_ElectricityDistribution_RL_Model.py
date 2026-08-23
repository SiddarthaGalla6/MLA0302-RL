'''Question: Develop a smart grid system uses Reinforcement Learning to manage electricity
distribution. Create an RL model by defining state space, action space, and reward
mechanism.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
DEMAND_BINS = 4
SUPPLY_BINS = 3
N_STATES = DEMAND_BINS * SUPPLY_BINS
ACTIONS = ['increase_supply', 'decrease_supply', 'load_shedding', 'maintain']
N_ACTIONS = len(ACTIONS)
def encode_state(demand, supply):
    return demand * SUPPLY_BINS + supply
def simulate_demand():
    return random.choices(range(DEMAND_BINS), weights=[0.3, 0.3, 0.25, 0.15])[0]
def compute_grid_reward(action, demand, supply):
    action_name = ACTIONS[action]
    if action_name == 'increase_supply':
        new_supply = min(supply + 1, SUPPLY_BINS - 1)
        cost = -1.5
    elif action_name == 'decrease_supply':
        new_supply = max(supply - 1, 0)
        cost = 1.0
    elif action_name == 'load_shedding':
        new_supply = supply
        cost = -6.0
    else:
        new_supply = supply
        cost = 0.0
    if demand > new_supply:
        reliability = -8.0 if action_name != 'load_shedding' else 3.0
    elif demand == new_supply:
        reliability = 4.0
    else:
        reliability = -1.0
    return cost + reliability, new_supply
def run_smartgrid_rl(n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, outages = [], []
    for ep in range(n_episodes):
        demand = simulate_demand()
        supply = 1
        total_reward, outage_count = 0, 0
        for step in range(24):
            state = encode_state(demand, supply)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            reward, new_supply = compute_grid_reward(action, demand, supply)
            if demand > new_supply:
                outage_count += 1
            new_demand = simulate_demand()
            next_state = encode_state(new_demand, new_supply)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            demand, supply = new_demand, new_supply
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        outages.append(outage_count)
    return ep_rewards, outages
print("Smart Grid Electricity Distribution RL Model")
print(f"State Space: demand_level x supply_level = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, outages = run_smartgrid_rl()
print(f"\nAvg reward first 50 episodes:  {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:   {np.mean(rewards[-50:]):.3f}")
print(f"Avg outages first 50 episodes: {np.mean(outages[:50]):.3f}")
print(f"Avg outages last 50 episodes:  {np.mean(outages[-50:]):.3f}")
print("\nRL Model Definition:")
print("  States : current electricity demand level x current supply level")
print("  Actions:", ACTIONS)
print("  Reward : supply adjustment cost + reliability score based on matching demand and supply")
print("  Objective: learn a dispatch policy that keeps supply aligned with demand and avoids outages")

'''
Output:
Smart Grid Electricity Distribution RL Model
State Space: demand_level x supply_level = 12
Actions: ['increase_supply', 'decrease_supply', 'load_shedding', 'maintain']

Avg reward first 50 episodes:  16.750
Avg reward last 50 episodes:   40.110
Avg outages first 50 episodes: 6.780
Avg outages last 50 episodes:  6.260

RL Model Definition:
  States : current electricity demand level x current supply level
  Actions: ['increase_supply', 'decrease_supply', 'load_shedding', 'maintain']
  Reward : supply adjustment cost + reliability score based on matching demand and supply
  Objective: learn a dispatch policy that keeps supply aligned with demand and avoids outages
'''
