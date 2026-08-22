'''Question: Justify the use of Reinforcement Learning in renewable energy management
systems. Define the RL framework components and discuss the impact of
environmental uncertainty on decision-making.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
SOLAR_BINS = 4
WIND_BINS = 4
BATTERY_BINS = 5
DEMAND_BINS = 3
N_STATES = SOLAR_BINS * WIND_BINS * BATTERY_BINS * DEMAND_BINS
ACTIONS = ['use_solar_wind', 'use_battery', 'draw_grid', 'store_surplus']
N_ACTIONS = len(ACTIONS)
def encode_state(solar, wind, battery, demand):
    return solar * WIND_BINS * BATTERY_BINS * DEMAND_BINS + wind * BATTERY_BINS * DEMAND_BINS + battery * DEMAND_BINS + demand
def simulate_weather():
    solar = random.choices(range(SOLAR_BINS), weights=[0.25, 0.25, 0.25, 0.25])[0]
    wind = random.choices(range(WIND_BINS), weights=[0.3, 0.3, 0.2, 0.2])[0]
    return solar, wind
def simulate_demand_level():
    return random.choices(range(DEMAND_BINS), weights=[0.4, 0.4, 0.2])[0]
def compute_energy_reward(action, solar, wind, battery, demand):
    action_name = ACTIONS[action]
    renewable_supply = solar + wind
    if action_name == 'use_solar_wind':
        if renewable_supply >= demand + 2:
            return 8.0
        else:
            return -5.0
    elif action_name == 'use_battery':
        if battery > 0:
            return 5.0
        else:
            return -10.0
    elif action_name == 'draw_grid':
        return -6.0
    else:
        surplus = renewable_supply - demand
        return 4.0 if surplus > 0 and battery < BATTERY_BINS - 1 else -2.0
def run_energy_rl(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, grid_draws = [], []
    for ep in range(n_episodes):
        solar, wind = simulate_weather()
        battery = random.randint(1, 3)
        demand = simulate_demand_level()
        total_reward, grid_count = 0, 0
        for step in range(24):
            state = encode_state(solar, wind, battery, demand)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            reward = compute_energy_reward(action, solar, wind, battery, demand)
            if ACTIONS[action] == 'use_battery' and battery > 0:
                battery -= 1
            elif ACTIONS[action] == 'store_surplus' and battery < BATTERY_BINS - 1:
                battery += 1
            elif ACTIONS[action] == 'draw_grid':
                grid_count += 1
            new_solar, new_wind = simulate_weather()
            new_demand = simulate_demand_level()
            next_state = encode_state(new_solar, new_wind, battery, new_demand)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            solar, wind, demand = new_solar, new_wind, new_demand
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        grid_draws.append(grid_count)
    return ep_rewards, grid_draws
print("Renewable Energy Management using Reinforcement Learning")
print(f"State Space: solar x wind x battery x demand = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, grid_draws = run_energy_rl()
print(f"\nAvg reward first 50 episodes:     {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:      {np.mean(rewards[-50:]):.3f}")
print(f"Avg grid draws first 50 episodes: {np.mean(grid_draws[:50]):.3f}")
print(f"Avg grid draws last 50 episodes:  {np.mean(grid_draws[-50:]):.3f}")
print("\nRL Framework Component Summary:")
print("  States : solar output level x wind output level x battery charge level x demand level")
print("  Actions:", ACTIONS)
print("  Reward : +renewable use, +battery use, -grid draw, +surplus storage when beneficial")
print("\nJustification and Environmental Uncertainty Impact:")
print("  Solar and wind output are stochastic and weather-dependent, unlike scriptable rule-based control")
print("  RL learns to balance battery storage against unpredictable renewable supply and demand swings")
print("  Reduced grid draw over training episodes shows adaptive decision-making under continued uncertainty")

'''
Output:
Renewable Energy Management using Reinforcement Learning
State Space: solar x wind x battery x demand = 240
Actions: ['use_solar_wind', 'use_battery', 'draw_grid', 'store_surplus']

Avg reward first 50 episodes:     93.420
Avg reward last 50 episodes:      138.920
Avg grid draws first 50 episodes: 1.760
Avg grid draws last 50 episodes:  0.340

RL Framework Component Summary:
  States : solar output level x wind output level x battery charge level x demand level
  Actions: ['use_solar_wind', 'use_battery', 'draw_grid', 'store_surplus']
  Reward : +renewable use, +battery use, -grid draw, +surplus storage when beneficial

Justification and Environmental Uncertainty Impact:
  Solar and wind output are stochastic and weather-dependent, unlike scriptable rule-based control
  RL learns to balance battery storage against unpredictable renewable supply and demand swings
  Reduced grid draw over training episodes shows adaptive decision-making under continued uncertainty
'''
