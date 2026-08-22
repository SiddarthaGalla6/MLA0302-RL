import numpy as np
np.random.seed(2)
N_STATES = 10
N_ACTIONS = 3
ACTIONS = ["move_fast", "move_slow", "stay"]
ENERGY_COST = {"move_fast": 3.0, "move_slow": 1.0, "stay": 0.2}
GOAL_STATE = 9
MAX_ENERGY = 20.0
def transition(state, action):
    if action == "move_fast":
        next_state = min(state + 2, GOAL_STATE)
    elif action == "move_slow":
        next_state = min(state + 1, GOAL_STATE)
    else:
        next_state = state
    return next_state
def reward_function(state, action, next_state, energy_used):
    cost = -ENERGY_COST[action]
    progress = 5.0 if next_state == GOAL_STATE else -0.5
    penalty = -10.0 if energy_used > MAX_ENERGY else 0.0
    return cost + progress + penalty
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.1, 0.95, 0.2, 300
for ep in range(episodes):
    state = 0
    energy_used = 0.0
    for step in range(20):
        action_idx = np.random.randint(N_ACTIONS) if np.random.rand() < epsilon else np.argmax(Q[state])
        action = ACTIONS[action_idx]
        next_state = transition(state, action)
        energy_used += ENERGY_COST[action]
        reward = reward_function(state, action, next_state, energy_used)
        Q[state, action_idx] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action_idx])
        state = next_state
        if state == GOAL_STATE or energy_used > MAX_ENERGY:
            break
print("MDP Components for Energy-Aware Robot Navigation")
print("States: position indices 0 to", GOAL_STATE)
print("Actions:", ACTIONS)
print("Energy Costs:", ENERGY_COST)
print("Energy Constraint: total energy per episode <=", MAX_ENERGY)
state = 0
energy_used = 0.0
path = [state]
for step in range(20):
    action_idx = np.argmax(Q[state])
    action = ACTIONS[action_idx]
    state = transition(state, action)
    energy_used += ENERGY_COST[action]
    path.append(state)
    if state == GOAL_STATE:
        break
print("Learned Path:", path)
print("Total Energy Used:", round(energy_used, 2))
