'''Question: Design and develop a Reinforcement Learning framework a ride-sharing company
uses Reinforcement Learning to match drivers with passengers efficiently. Apply RL
concepts to define the state space, action space, and reward function to minimize
waiting time and improve service quality.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_ZONES = 5
DRIVER_BINS = 3
REQUEST_BINS = 3
N_STATES = N_ZONES * DRIVER_BINS * REQUEST_BINS
ACTIONS = ['match_nearest', 'match_highest_rated', 'wait_for_better', 'reposition']
N_ACTIONS = len(ACTIONS)
def encode_state(zone, drivers, requests):
    return zone * DRIVER_BINS * REQUEST_BINS + drivers * REQUEST_BINS + requests
def simulate_zone_conditions():
    drivers = random.choices(range(DRIVER_BINS), weights=[0.3, 0.4, 0.3])[0]
    requests = random.choices(range(REQUEST_BINS), weights=[0.3, 0.4, 0.3])[0]
    return drivers, requests
def compute_matching_reward(action, drivers, requests, wait_time):
    action_name = ACTIONS[action]
    if action_name == 'match_nearest':
        service_quality = 3.0
        wait_penalty = -wait_time * 0.3
        supply_bonus = 2.0 if drivers >= requests else -3.0
    elif action_name == 'match_highest_rated':
        service_quality = 5.0
        wait_penalty = -wait_time * 0.6
        supply_bonus = 1.0 if drivers >= requests else -3.0
    elif action_name == 'wait_for_better':
        service_quality = -1.0
        wait_penalty = -wait_time * 1.0
        supply_bonus = 0.0
    else:
        service_quality = 0.5
        wait_penalty = 0.0
        supply_bonus = 2.0 if drivers < requests else -1.0
    return service_quality + wait_penalty + supply_bonus
def run_ridesharing_rl(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.12, 0.9, 0.3
    ep_rewards, wait_times = [], []
    for ep in range(n_episodes):
        zone = random.randint(0, N_ZONES - 1)
        drivers, requests = simulate_zone_conditions()
        wait_time = 0
        total_reward, total_wait = 0, 0
        for step in range(15):
            state = encode_state(zone, drivers, requests)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            if ACTIONS[action] in ['match_nearest', 'match_highest_rated']:
                wait_time = max(wait_time - 2, 0)
            else:
                wait_time += 1
            reward = compute_matching_reward(action, drivers, requests, wait_time)
            zone = random.randint(0, N_ZONES - 1)
            drivers, requests = simulate_zone_conditions()
            next_state = encode_state(zone, drivers, requests)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            total_reward += reward
            total_wait += wait_time
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        wait_times.append(total_wait / 15)
    return ep_rewards, wait_times
print("Ride-Sharing Driver-Passenger Matching using Reinforcement Learning")
print(f"State Space: zone x available_drivers x pending_requests = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, wait_times = run_ridesharing_rl()
print(f"\nAvg reward first 50 episodes:    {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:     {np.mean(rewards[-50:]):.3f}")
print(f"Avg wait time first 50 episodes: {np.mean(wait_times[:50]):.3f}")
print(f"Avg wait time last 50 episodes:  {np.mean(wait_times[-50:]):.3f}")
print("\nRL Framework Component Summary:")
print("  States : city zone x driver availability level x passenger request level")
print("  Actions:", ACTIONS)
print("  Reward : service quality score - waiting time penalty + supply-demand balance bonus")
print("  Goal   : learn a matching policy that minimizes passenger wait while keeping service quality high")

'''
Output:
Ride-Sharing Driver-Passenger Matching using Reinforcement Learning
State Space: zone x available_drivers x pending_requests = 45
Actions: ['match_nearest', 'match_highest_rated', 'wait_for_better', 'reposition']

Avg reward first 50 episodes:    49.420
Avg reward last 50 episodes:     56.736
Avg wait time first 50 episodes: 0.264
Avg wait time last 50 episodes:  0.228

RL Framework Component Summary:
  States : city zone x driver availability level x passenger request level
  Actions: ['match_nearest', 'match_highest_rated', 'wait_for_better', 'reposition']
  Reward : service quality score - waiting time penalty + supply-demand balance bonus
  Goal   : learn a matching policy that minimizes passenger wait while keeping service quality high
'''
