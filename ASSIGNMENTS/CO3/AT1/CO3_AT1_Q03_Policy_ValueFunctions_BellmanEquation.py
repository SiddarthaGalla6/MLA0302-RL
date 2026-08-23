'''Question: Discuss the role of policy and value functions in RL, and examine how state-value
and action-value functions contribute to evaluating long-term benefits of actions using
the Bellman Equation.'''

# Code:
import numpy as np
np.random.seed(42)
N_STATES = 5
N_ACTIONS = 2
GOAL = 4
gamma = 0.9
def transition(state, action):
    if action == 0:
        return max(state - 1, 0)
    return min(state + 1, N_STATES - 1)
def reward(state, action, next_state):
    if next_state == GOAL:
        return 10.0
    if next_state == state:
        return -1.0
    return -0.2
policy = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
def evaluate_state_value(policy, theta=1e-4, max_iter=200):
    V = np.zeros(N_STATES)
    for iteration in range(max_iter):
        delta = 0.0
        new_V = V.copy()
        for s in range(N_STATES):
            a = policy[s]
            s_next = transition(s, a)
            r = reward(s, a, s_next)
            new_V[s] = r + gamma * V[s_next]
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    return V, iteration
def compute_action_value(V):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s_next = transition(s, a)
            r = reward(s, a, s_next)
            Q[s, a] = r + gamma * V[s_next]
    return Q
V, iterations = evaluate_state_value(policy)
Q = compute_action_value(V)
print("Policy and Value Functions with Bellman Equation")
print(f"State Space: {N_STATES}, Action Space: 0=left, 1=right, Goal state: {GOAL}")
print(f"Policy under evaluation: {policy}")
print(f"Converged after {iterations} sweeps\n")
print(f"{'State':>6} {'V(s)':>10} {'Q(s,left)':>11} {'Q(s,right)':>12}")
print("-" * 42)
for s in range(N_STATES):
    print(f"{s:>6} {V[s]:>10.3f} {Q[s, 0]:>11.3f} {Q[s, 1]:>12.3f}")
print("\nBellman Equation Used:")
print("  V_pi(s) = R(s, pi(s), s') + gamma * V_pi(s')")
print("  Q_pi(s, a) = R(s, a, s') + gamma * V_pi(s')")
print("\nRole of Policy and Value Functions:")
print("  Policy pi(s) defines which action the agent takes in each state")
print("  State-value V(s) evaluates how good it is to be in a state under the current policy")
print("  Action-value Q(s,a) evaluates how good a specific action is in a state, guiding policy improvement")
print("  Comparing Q(s,left) vs Q(s,right) at each state shows whether the current policy is already optimal there")

'''
Output:
Policy and Value Functions with Bellman Equation
State Space: 5, Action Space: 0=left, 1=right, Goal state: 4
Policy under evaluation: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
Converged after 110 sweeps

 State       V(s)   Q(s,left)   Q(s,right)
------------------------------------------
     0     72.357      64.121       72.357
     1     80.619      64.921       80.619
     2     89.799      72.357       89.799
     3     99.999      80.619       99.999
     4     99.999      89.799       99.999

Bellman Equation Used:
  V_pi(s) = R(s, pi(s), s') + gamma * V_pi(s')
  Q_pi(s, a) = R(s, a, s') + gamma * V_pi(s')

Role of Policy and Value Functions:
  Policy pi(s) defines which action the agent takes in each state
  State-value V(s) evaluates how good it is to be in a state under the current policy
  Action-value Q(s,a) evaluates how good a specific action is in a state, guiding policy improvement
  Comparing Q(s,left) vs Q(s,right) at each state shows whether the current policy is already optimal there
'''
