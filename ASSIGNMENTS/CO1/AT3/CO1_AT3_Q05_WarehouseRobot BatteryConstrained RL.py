import numpy as np
np.random.seed(4)
N_LOCATIONS = 8
BATTERY_LIMIT = 30
DELIVERY_TIME = 4
RETURN_TIME = 2
N_STATES = (BATTERY_LIMIT + 1) * N_LOCATIONS
N_ACTIONS = N_LOCATIONS
def state_id(battery, location):
    return battery * N_LOCATIONS + location
def reward_function(battery_before, battery_after, delivered):
    if battery_after < 0:
        return -50.0
    reward = 10.0 if delivered else -1.0
    reward -= 0.1 * (battery_before - battery_after)
    return reward
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 400
for ep in range(episodes):
    battery = BATTERY_LIMIT
    location = 0
    deliveries = 0
    for step in range(30):
        state = state_id(battery, location)
        action = np.random.randint(N_ACTIONS) if np.random.rand() < epsilon else np.argmax(Q[state])
        target = action
        cost = DELIVERY_TIME + RETURN_TIME
        battery_before = battery
        battery_after = battery - cost
        delivered = battery_after >= 0 and target != location
        if delivered:
            deliveries += 1
            battery = max(battery_after, 0)
            location = 0
        reward = reward_function(battery_before, battery_after, delivered)
        next_state = state_id(battery, location)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        if battery <= 0:
            break
print("Warehouse Robot RL under Battery Constraint")
print("Battery capacity (minutes):", BATTERY_LIMIT)
print("Delivery+Return time cost per trip:", DELIVERY_TIME + RETURN_TIME)
print("Locations:", N_LOCATIONS)
battery = BATTERY_LIMIT
location = 0
deliveries = 0
for step in range(30):
    state = state_id(battery, location)
    action = np.argmax(Q[state])
    cost = DELIVERY_TIME + RETURN_TIME
    if battery - cost < 0:
        break
    battery -= cost
    deliveries += 1
    location = 0
print("Max deliveries completed within battery limit:", deliveries)
print("Remaining battery:", battery)
