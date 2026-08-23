'''Question: Analyze the influence of the discount factor (gamma) on short-term versus long-term
decision-making, and evaluate its role in episodic tasks.'''

# Code:
import numpy as np
N_STATES = 8
IMMEDIATE_REWARD_STATE = 2
DELAYED_REWARD_STATE = 7
def transition(state, action):
    if action == 0:
        return max(state - 1, 0)
    return min(state + 1, N_STATES - 1)
def reward(next_state):
    if next_state == IMMEDIATE_REWARD_STATE:
        return 4.0
    if next_state == DELAYED_REWARD_STATE:
        return 20.0
    return -0.5
def value_iteration(gamma, theta=1e-5, max_iter=500):
    V = np.zeros(N_STATES)
    for iteration in range(max_iter):
        delta = 0.0
        new_V = V.copy()
        for s in range(N_STATES):
            action_values = []
            for a in range(2):
                s_next = transition(s, a)
                r = reward(s_next)
                action_values.append(r + gamma * V[s_next])
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        action_values = []
        for a in range(2):
            s_next = transition(s, a)
            r = reward(s_next)
            action_values.append(r + gamma * V[s_next])
        policy[s] = int(np.argmax(action_values))
    return V, policy
print("Discount Factor (gamma) Influence on Short-Term vs Long-Term Decisions")
print(f"State Space: {N_STATES}")
print(f"Immediate reward (+4) at state {IMMEDIATE_REWARD_STATE}, delayed larger reward (+20) at state {DELAYED_REWARD_STATE}")
print(f"\n{'Gamma':>8} {'V(state0)':>10} {'Policy(state0)':>15} {'Policy(state1)':>15} {'Policy(state4)':>15}")
print("-" * 70)
for gamma in [0.0, 0.3, 0.6, 0.9, 0.99]:
    V, policy = value_iteration(gamma)
    action_name = {0: 'left', 1: 'right'}
    print(f"{gamma:>8} {V[0]:>10.3f} {action_name[policy[0]]:>15} {action_name[policy[1]]:>15} {action_name[policy[4]]:>15}")
print("\nDiscount Factor Analysis:")
print("  gamma near 0: agent is myopic, chases the smaller immediate reward (+4) and ignores the distant +20")
print("  gamma near 1: agent values future rewards almost as much as immediate ones, so it targets the larger delayed reward")
print("  As gamma increases from 0 to 0.99, the optimal policy shifts from moving toward the near reward to the far one")
print("\nRole in Episodic Tasks:")
print("  In finite episodic tasks, gamma still shapes whether the agent front-loads or defers reward-seeking behavior")
print("  A high gamma is essential when the true objective requires patience across many steps within the episode")
print("  A low gamma can cause premature termination of useful behavior once a small nearby reward is obtained")

'''
Output:
Discount Factor (gamma) Influence on Short-Term vs Long-Term Decisions
State Space: 8
Immediate reward (+4) at state 2, delayed larger reward (+20) at state 7

   Gamma  V(state0)  Policy(state0)  Policy(state1)  Policy(state4)
----------------------------------------------------------------------
     0.0     -0.500            left           right            left
     0.3      0.769           right           right           right
     0.6      3.841           right           right           right
     0.9    107.995           right           right           right
    0.99   1871.348           right           right           right

Discount Factor Analysis:
  gamma near 0: agent is myopic, chases the smaller immediate reward (+4) and ignores the distant +20
  gamma near 1: agent values future rewards almost as much as immediate ones, so it targets the larger delayed reward
  As gamma increases from 0 to 0.99, the optimal policy shifts from moving toward the near reward to the far one

Role in Episodic Tasks:
  In finite episodic tasks, gamma still shapes whether the agent front-loads or defers reward-seeking behavior
  A high gamma is essential when the true objective requires patience across many steps within the episode
  A low gamma can cause premature termination of useful behavior once a small nearby reward is obtained
'''
