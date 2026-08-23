'''Question: Differentiate between on-policy and off-policy learning methods in RL, and assess
their advantages and limitations in practical applications.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
GRID_WIDTH, GRID_HEIGHT = 6, 3
N_STATES = GRID_WIDTH * GRID_HEIGHT
N_ACTIONS = 4
CLIFF = [(1, x) for x in range(1, 5)]
START = (2, 0)
GOAL = (2, 5)
def state_id(r, c):
    return r * GRID_WIDTH + c
def step_env(r, c, action):
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    dr, dc = moves[action]
    nr, nc = min(max(r + dr, 0), GRID_HEIGHT - 1), min(max(c + dc, 0), GRID_WIDTH - 1)
    if (nr, nc) in CLIFF:
        return START[0], START[1], -100.0, True
    if (nr, nc) == GOAL:
        return nr, nc, 10.0, True
    return nr, nc, -1.0, False
def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(np.argmax(Q[state]))
def run_sarsa(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.95, 0.2
    ep_returns = []
    for ep in range(n_episodes):
        r, c = START
        action = choose_action(state_id(r, c), Q, epsilon)
        total_reward = 0
        for step in range(100):
            nr, nc, reward, done = step_env(r, c, action)
            next_state = state_id(nr, nc)
            next_action = choose_action(next_state, Q, epsilon)
            state = state_id(r, c)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            r, c, action = nr, nc, next_action
            total_reward += reward
            if done and (nr, nc) == GOAL:
                break
            if done:
                r, c = START
                action = choose_action(state_id(r, c), Q, epsilon)
        ep_returns.append(total_reward)
    return ep_returns, Q
def run_qlearning(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.95, 0.2
    ep_returns = []
    for ep in range(n_episodes):
        r, c = START
        total_reward = 0
        for step in range(100):
            state = state_id(r, c)
            action = choose_action(state, Q, epsilon)
            nr, nc, reward, done = step_env(r, c, action)
            next_state = state_id(nr, nc)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            r, c = nr, nc
            total_reward += reward
            if done and (nr, nc) == GOAL:
                break
            if done:
                r, c = START
        ep_returns.append(total_reward)
    return ep_returns, Q
print("On-Policy (SARSA) vs Off-Policy (Q-learning) on a Cliff-Walking Task")
print(f"Grid: {GRID_HEIGHT}x{GRID_WIDTH}, Start: {START}, Goal: {GOAL}, Cliff cells: {CLIFF}")
sarsa_returns, Q_sarsa = run_sarsa()
qlearn_returns, Q_qlearn = run_qlearning()
print(f"\n{'Method':>12} {'AvgReturnFirst50':>17} {'AvgReturnLast50':>16} {'FellOffCliffLast50':>19}")
print("-" * 68)
sarsa_falls = sum(1 for r in sarsa_returns[-50:] if r <= -100)
qlearn_falls = sum(1 for r in qlearn_returns[-50:] if r <= -100)
print(f"{'SARSA':>12} {np.mean(sarsa_returns[:50]):>17.2f} {np.mean(sarsa_returns[-50:]):>16.2f} {sarsa_falls:>19}")
print(f"{'Q-learning':>12} {np.mean(qlearn_returns[:50]):>17.2f} {np.mean(qlearn_returns[-50:]):>16.2f} {qlearn_falls:>19}")
print("\nOn-Policy vs Off-Policy Differentiation:")
print("  SARSA (on-policy) updates using the action actually taken by the behavior policy, including exploratory moves")
print("  Q-learning (off-policy) updates using the greedy best next action, regardless of what action is actually taken next")
print("  In this specific run Q-learning converged to fewer cliff falls and a better return than SARSA by the last 50 episodes")
print("  The relative safety of each method in practice depends on the epsilon schedule, environment layout, and training budget")
print("\nAdvantages and Limitations:")
print("  On-policy: value estimates match the policy actually being followed, useful when training-time behavior itself matters")
print("  Off-policy: can learn an optimal target policy from a different, more exploratory behavior policy, and can reuse past data")
print("  Off-policy methods risk instability with function approximation; on-policy methods can be sample-inefficient when reused")

'''
Output:
On-Policy (SARSA) vs Off-Policy (Q-learning) on a Cliff-Walking Task
Grid: 3x6, Start: (2, 0), Goal: (2, 5), Cliff cells: [(1, 1), (1, 2), (1, 3), (1, 4)]

      Method  AvgReturnFirst50  AvgReturnLast50  FellOffCliffLast50
--------------------------------------------------------------------
       SARSA           -163.22           -75.00                  17
  Q-learning            -76.92           -13.00                   2

On-Policy vs Off-Policy Differentiation:
  SARSA (on-policy) updates using the action actually taken by the behavior policy, including exploratory moves
  Q-learning (off-policy) updates using the greedy best next action, regardless of what action is actually taken next
  In this specific run Q-learning converged to fewer cliff falls and a better return than SARSA by the last 50 episodes
  The relative safety of each method in practice depends on the epsilon schedule, environment layout, and training budget

Advantages and Limitations:
  On-policy: value estimates match the policy actually being followed, useful when training-time behavior itself matters
  Off-policy: can learn an optimal target policy from a different, more exploratory behavior policy, and can reuse past data
  Off-policy methods risk instability with function approximation; on-policy methods can be sample-inefficient when reused
'''
