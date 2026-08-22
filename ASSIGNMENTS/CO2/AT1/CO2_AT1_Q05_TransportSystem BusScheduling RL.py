'''Question: Analyze how Reinforcement Learning can be used to optimize transportation
systems such as bus scheduling or route allocation. Define states, actions, and
rewards, and discuss how real-time data improves performance.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_STOPS = 6
DEMAND_BINS = 4
N_STATES = N_STOPS * DEMAND_BINS * DEMAND_BINS
ACTIONS = ['dispatch_now', 'wait_short', 'wait_long', 'reroute']
N_ACTIONS = len(ACTIONS)
def encode_state(stop, demand, congestion):
    return stop * DEMAND_BINS * DEMAND_BINS + demand * DEMAND_BINS + congestion
def simulate_demand(stop, hour_bucket):
    peak = hour_bucket in [1, 3]
    weights = [0.1, 0.2, 0.3, 0.4] if peak else [0.5, 0.3, 0.15, 0.05]
    return random.choices(range(DEMAND_BINS), weights=weights)[0]
def simulate_congestion():
    return random.choices(range(DEMAND_BINS), weights=[0.4, 0.3, 0.2, 0.1])[0]
def compute_bus_reward(action, demand, congestion, wait_time):
    action_name = ACTIONS[action]
    passenger_satisfaction = 0.0
    if action_name == 'dispatch_now':
        passenger_satisfaction = demand * 3.0 - wait_time * 0.5
    elif action_name == 'wait_short':
        passenger_satisfaction = -1.0 if demand >= 2 else 1.0
    elif action_name == 'wait_long':
        passenger_satisfaction = -4.0 if demand >= 2 else 0.5
    else:
        passenger_satisfaction = 2.0 if congestion >= 2 else -2.0
    congestion_penalty = -congestion * 0.5 if action_name == 'dispatch_now' else 0.0
    return passenger_satisfaction + congestion_penalty
def run_bus_scheduling_rl(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, avg_wait = [], []
    for ep in range(n_episodes):
        stop = random.randint(0, N_STOPS - 1)
        hour_bucket = random.randint(0, 3)
        wait_time = 0
        total_reward, total_wait = 0, 0
        for step in range(20):
            demand = simulate_demand(stop, hour_bucket)
            congestion = simulate_congestion()
            state = encode_state(stop, demand, congestion)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            if ACTIONS[action] == 'dispatch_now':
                wait_time = 0
            else:
                wait_time += 1
            reward = compute_bus_reward(action, demand, congestion, wait_time)
            stop = (stop + 1) % N_STOPS
            hour_bucket = (hour_bucket + (1 if random.random() < 0.15 else 0)) % 4
            next_demand = simulate_demand(stop, hour_bucket)
            next_congestion = simulate_congestion()
            next_state = encode_state(stop, next_demand, next_congestion)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            total_reward += reward
            total_wait += wait_time
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        avg_wait.append(total_wait / 20)
    return ep_rewards, avg_wait
print("Bus Scheduling and Route Allocation using Reinforcement Learning")
print(f"State Space: stop x demand_level x congestion_level = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, avg_wait = run_bus_scheduling_rl()
print(f"\nAvg reward first 50 episodes:  {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:   {np.mean(rewards[-50:]):.3f}")
print(f"Avg passenger wait first 50:   {np.mean(avg_wait[:50]):.3f}")
print(f"Avg passenger wait last 50:    {np.mean(avg_wait[-50:]):.3f}")
print("\nMDP Component Summary:")
print("  States : current stop x passenger demand level x traffic congestion level")
print("  Actions:", ACTIONS)
print("  Reward : passenger_satisfaction - congestion_penalty, tuned to minimize wait time")
print("\nReal-Time Data Impact:")
print("  Live demand and congestion readings each step let the policy dispatch buses adaptively")
print("  Without real-time data the agent would rely on fixed timetables, ignoring surges in demand")
print("  Continual retraining on streaming ridership data keeps the schedule aligned with actual patterns")

'''
Output:
Bus Scheduling and Route Allocation using Reinforcement Learning
State Space: stop x demand_level x congestion_level = 96
Actions: ['dispatch_now', 'wait_short', 'wait_long', 'reroute']

Avg reward first 50 episodes:  56.690
Avg reward last 50 episodes:   72.160
Avg passenger wait first 50:   0.630
Avg passenger wait last 50:    0.463

MDP Component Summary:
  States : current stop x passenger demand level x traffic congestion level
  Actions: ['dispatch_now', 'wait_short', 'wait_long', 'reroute']
  Reward : passenger_satisfaction - congestion_penalty, tuned to minimize wait time

Real-Time Data Impact:
  Live demand and congestion readings each step let the policy dispatch buses adaptively
  Without real-time data the agent would rely on fixed timetables, ignoring surges in demand
  Continual retraining on streaming ridership data keeps the schedule aligned with actual patterns
'''
