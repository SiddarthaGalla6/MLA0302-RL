import numpy as np
np.random.seed(8)
GRID_SIZE = 6
N_ACTIONS = 4
BATTERY_MAX = 25
NO_FLY_ZONES = [(2, 2), (2, 3), (3, 2), (4, 4)]
DELIVERY_POINTS = [(5, 5), (0, 5), (5, 0)]
def state_id(r, c, battery):
    return (r * GRID_SIZE + c) * (BATTERY_MAX + 1) + battery
N_STATES = GRID_SIZE * GRID_SIZE * (BATTERY_MAX + 1)
moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.25, 500
def bellman_target(reward, next_state):
    return reward + gamma * np.max(Q[next_state])
for ep in range(episodes):
    r, c, battery = 0, 0, BATTERY_MAX
    deliveries = 0
    for step in range(40):
        state = state_id(r, c, battery)
        action = np.random.randint(N_ACTIONS) if np.random.rand() < epsilon else np.argmax(Q[state])
        dr, dc = moves[action]
        nr, nc = min(max(r + dr, 0), GRID_SIZE - 1), min(max(c + dc, 0), GRID_SIZE - 1)
        battery = max(battery - 1, 0)
        if (nr, nc) in NO_FLY_ZONES:
            reward = -20.0
            nr, nc = r, c
        elif (nr, nc) in DELIVERY_POINTS:
            reward = 15.0
            deliveries += 1
        elif battery == 0:
            reward = -15.0
        else:
            reward = -0.5
        r, c = nr, nc
        next_state = state_id(r, c, battery)
        Q[state, action] += alpha * (bellman_target(reward, next_state) - Q[state, action])
        if battery == 0:
            break
print("Delivery Drone RL Framework")
print("Grid size:", GRID_SIZE, "x", GRID_SIZE, "Battery max:", BATTERY_MAX)
print("No-fly zones:", NO_FLY_ZONES)
print("Delivery points:", DELIVERY_POINTS)
r, c, battery = 0, 0, BATTERY_MAX
deliveries = 0
for step in range(40):
    state = state_id(r, c, battery)
    action = np.argmax(Q[state])
    dr, dc = moves[action]
    r, c = min(max(r + dr, 0), GRID_SIZE - 1), min(max(c + dc, 0), GRID_SIZE - 1)
    battery = max(battery - 1, 0)
    if (r, c) in DELIVERY_POINTS:
        deliveries += 1
    if battery == 0:
        break
print("Deliveries completed by learned policy:", deliveries)
print("Battery remaining:", battery)
