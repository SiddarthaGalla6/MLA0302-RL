import numpy as np
np.random.seed(9)
LOAD_BINS = 5
N_ACTIONS = 3
ACTIONS = ["run_appliance", "delay_appliance", "use_battery"]
THRESHOLD = 15.0
N_STATES = LOAD_BINS * 2
def state_id(load_bin, over_threshold):
    return load_bin * 2 + over_threshold
def reward_function(action, current_load, threshold):
    if action == "run_appliance":
        new_load = current_load + 3.0
        efficiency = -1.0
    elif action == "delay_appliance":
        new_load = current_load
        efficiency = 1.0
    else:
        new_load = current_load + 1.0
        efficiency = 0.5
    feasibility_penalty = -15.0 if new_load > threshold else 5.0
    shaping = -0.5 * max(new_load - threshold, 0)
    return efficiency + feasibility_penalty + shaping, new_load
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 400
for ep in range(episodes):
    current_load = np.random.uniform(5, 10)
    for step in range(24):
        load_bin = min(int(current_load / 4), LOAD_BINS - 1)
        over = 1 if current_load > THRESHOLD else 0
        state = state_id(load_bin, over)
        action_idx = np.random.randint(N_ACTIONS) if np.random.rand() < epsilon else np.argmax(Q[state])
        reward, current_load = reward_function(ACTIONS[action_idx], current_load, THRESHOLD)
        current_load = max(current_load - np.random.uniform(1, 3), 0)
        next_load_bin = min(int(current_load / 4), LOAD_BINS - 1)
        next_over = 1 if current_load > THRESHOLD else 0
        next_state = state_id(next_load_bin, next_over)
        Q[state, action_idx] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action_idx])
    epsilon = max(0.05, epsilon * 0.98)
print("Smart Energy Management RL under Consumption Threshold")
print("Consumption threshold (kWh):", THRESHOLD)
print("Actions:", ACTIONS)
print("States: load_bin x over_threshold_flag =", N_STATES)
current_load = 8.0
total_violations = 0
for step in range(24):
    load_bin = min(int(current_load / 4), LOAD_BINS - 1)
    over = 1 if current_load > THRESHOLD else 0
    state = state_id(load_bin, over)
    action_idx = np.argmax(Q[state])
    reward, current_load = reward_function(ACTIONS[action_idx], current_load, THRESHOLD)
    if current_load > THRESHOLD:
        total_violations += 1
    current_load = max(current_load - np.random.uniform(1, 3), 0)
print("Threshold violations over 24-hour test run:", total_violations)
print("Final load level:", round(current_load, 2))
