import numpy as np
np.random.seed(5)
QUEUE_BINS = 5
N_STATES = QUEUE_BINS * QUEUE_BINS * 2
N_ACTIONS = 2
ACTIONS = ["NS_green", "EW_green"]
def state_id(ns_queue, ew_queue, emergency):
    return ns_queue * QUEUE_BINS * 2 + ew_queue * 2 + emergency
def reward_function(ns_queue, ew_queue, action, emergency, emergency_lane):
    if emergency:
        if (emergency_lane == 0 and action == 0) or (emergency_lane == 1 and action == 1):
            return 20.0
        else:
            return -30.0
    wait_penalty = -(ns_queue + ew_queue)
    served = ns_queue if action == 0 else ew_queue
    return wait_penalty + served
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.3, 400
for ep in range(episodes):
    ns_queue = np.random.randint(QUEUE_BINS)
    ew_queue = np.random.randint(QUEUE_BINS)
    for step in range(20):
        emergency = 1 if np.random.rand() < 0.1 else 0
        emergency_lane = np.random.randint(2) if emergency else -1
        state = state_id(ns_queue, ew_queue, emergency)
        if emergency:
            action = emergency_lane
        elif np.random.rand() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = np.argmax(Q[state])
        reward = reward_function(ns_queue, ew_queue, action, emergency, emergency_lane)
        if action == 0:
            ns_queue = max(ns_queue - 2, 0)
            ew_queue = min(ew_queue + 1, QUEUE_BINS - 1)
        else:
            ew_queue = max(ew_queue - 2, 0)
            ns_queue = min(ns_queue + 1, QUEUE_BINS - 1)
        next_state = state_id(ns_queue, ew_queue, 0)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    epsilon = max(0.05, epsilon * 0.98)
print("Traffic Signal RL with Emergency Vehicle Priority")
print("States: NS_queue x EW_queue x emergency_flag =", N_STATES)
print("Actions:", ACTIONS)
print("Safety Constraint: emergency lane always forced green")
test_state = state_id(3, 2, 1)
print("Q-values in emergency state (NS=3,EW=2,emergency=1):", Q[test_state])
print("Chosen action under emergency override: forced to emergency lane, bypassing epsilon-greedy exploration")
