'''Question: A smart irrigation system uses Reinforcement Learning to optimize water usage.
Create an RL framework by defining states, actions, and reward function.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
SOIL_MOISTURE_BINS = 4
WEATHER_BINS = 3
CROP_STAGE_BINS = 3
N_STATES = SOIL_MOISTURE_BINS * WEATHER_BINS * CROP_STAGE_BINS
ACTIONS = ['irrigate_heavy', 'irrigate_light', 'no_irrigation']
N_ACTIONS = len(ACTIONS)
OPTIMAL_MOISTURE = 2
def encode_state(moisture, weather, crop_stage):
    return moisture * WEATHER_BINS * CROP_STAGE_BINS + weather * CROP_STAGE_BINS + crop_stage
def simulate_weather():
    return random.choices(range(WEATHER_BINS), weights=[0.5, 0.3, 0.2])[0]
def compute_irrigation_reward(action, moisture, weather, water_used):
    action_name = ACTIONS[action]
    deviation = abs(moisture - OPTIMAL_MOISTURE)
    crop_health = 5.0 - deviation * 2.0
    water_penalty = -water_used * 0.3
    if weather == 2 and action_name != 'no_irrigation':
        rain_waste_penalty = -4.0
    else:
        rain_waste_penalty = 0.0
    return crop_health + water_penalty + rain_waste_penalty
def run_irrigation_rl(n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, water_usage = [], []
    for ep in range(n_episodes):
        moisture = random.randint(0, SOIL_MOISTURE_BINS - 1)
        weather = simulate_weather()
        crop_stage = random.randint(0, CROP_STAGE_BINS - 1)
        total_reward, total_water = 0, 0
        for step in range(20):
            state = encode_state(moisture, weather, crop_stage)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            action_name = ACTIONS[action]
            if action_name == 'irrigate_heavy':
                water_used = 3
                moisture = min(moisture + 2, SOIL_MOISTURE_BINS - 1)
            elif action_name == 'irrigate_light':
                water_used = 1
                moisture = min(moisture + 1, SOIL_MOISTURE_BINS - 1)
            else:
                water_used = 0
            if weather == 2:
                moisture = min(moisture + 1, SOIL_MOISTURE_BINS - 1)
            else:
                moisture = max(moisture - 1, 0)
            reward = compute_irrigation_reward(action, moisture, weather, water_used)
            weather = simulate_weather()
            crop_stage = min(crop_stage + (1 if step % 7 == 0 else 0), CROP_STAGE_BINS - 1)
            next_state = encode_state(moisture, weather, crop_stage)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            total_reward += reward
            total_water += water_used
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        water_usage.append(total_water)
    return ep_rewards, water_usage
print("Smart Irrigation Water Usage Optimization using Reinforcement Learning")
print(f"State Space: soil_moisture x weather x crop_stage = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, water_usage = run_irrigation_rl()
print(f"\nAvg reward first 50 episodes:      {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:       {np.mean(rewards[-50:]):.3f}")
print(f"Avg water used first 50 episodes:  {np.mean(water_usage[:50]):.3f}")
print(f"Avg water used last 50 episodes:   {np.mean(water_usage[-50:]):.3f}")
print("\nRL Framework Definition:")
print("  States : soil moisture level x current weather condition x crop growth stage")
print("  Actions:", ACTIONS)
print("  Reward : crop health score based on moisture deviation - water usage penalty - rain-waste penalty")
print("  Outcome: policy learns to skip irrigation before rain and apply water only when soil is genuinely dry")

'''
Output:
Smart Irrigation Water Usage Optimization using Reinforcement Learning
State Space: soil_moisture x weather x crop_stage = 36
Actions: ['irrigate_heavy', 'irrigate_light', 'no_irrigation']

Avg reward first 50 episodes:      70.396
Avg reward last 50 episodes:       75.994
Avg water used first 50 episodes:  39.880
Avg water used last 50 episodes:   28.820

RL Framework Definition:
  States : soil moisture level x current weather condition x crop growth stage
  Actions: ['irrigate_heavy', 'irrigate_light', 'no_irrigation']
  Reward : crop health score based on moisture deviation - water usage penalty - rain-waste penalty
  Outcome: policy learns to skip irrigation before rain and apply water only when soil is genuinely dry
'''
