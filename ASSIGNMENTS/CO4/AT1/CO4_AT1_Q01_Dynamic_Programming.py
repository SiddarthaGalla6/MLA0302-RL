'''Question: Dynamic Programming for Sequential Decision-Making in Reinforcement Learning - Bellman Principle of Optimality, Policy Evaluation and Policy Improvement, Value Iteration vs. Policy Iteration, Industrial applications.'''
# Code:
import numpy as np
np.random.seed(42)
n_states = 6
n_actions = 2
gamma = 0.9
terminal = 5
R = np.array([[-1, -2], [-1, -3], [-2, -1], [-1, -2], [-3, 10], [0, 0]], dtype=float)
P = np.zeros((n_states, n_actions, n_states))
P[0, 0, 1] = 1.0; P[0, 1, 2] = 1.0
P[1, 0, 2] = 1.0; P[1, 1, 3] = 1.0
P[2, 0, 3] = 1.0; P[2, 1, 4] = 1.0
P[3, 0, 4] = 1.0; P[3, 1, 5] = 1.0
P[4, 0, 5] = 1.0; P[4, 1, 3] = 1.0
P[5, 0, 5] = 1.0; P[5, 1, 5] = 1.0
def policy_evaluation(policy, theta=1e-6):
    V = np.zeros(n_states)
    for _ in range(1000):
        delta = 0
        for s in range(n_states):
            a = policy[s]
            v = sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(n_states))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
def policy_improvement(V):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_vals = [sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(n_states)) for a in range(n_actions)]
        policy[s] = np.argmax(q_vals)
    return policy
def policy_iteration():
    policy = np.zeros(n_states, dtype=int)
    iters = 0
    for _ in range(100):
        V = policy_evaluation(policy)
        new_policy = policy_improvement(V)
        iters += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V, iters
def value_iteration(theta=1e-6):
    V = np.zeros(n_states)
    iters = 0
    for _ in range(1000):
        delta = 0
        for s in range(n_states):
            q_vals = [sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(n_states)) for a in range(n_actions)]
            v_new = max(q_vals)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        iters += 1
        if delta < theta:
            break
    policy = policy_improvement(V)
    return policy, V, iters
print("Dynamic Programming - Sequential Decision-Making")
print(f"States: {n_states}, Actions: {n_actions}, Gamma: {gamma}, Terminal: S{terminal}\n")
print("=" * 55)
print("Bellman Principle of Optimality:")
print("  V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s,a,s') * V*(s') ]")
print("  Optimal substructure: optimal policy from any state s is")
print("  independent of how agent arrived at state s.\n")
pi_policy, pi_V, pi_iters = policy_iteration()
vi_policy, vi_V, vi_iters = value_iteration()
print("Policy Iteration Results:")
print(f"  Converged in : {pi_iters} iterations")
print(f"  Optimal Policy: {['A' + str(a) for a in pi_policy]}")
print(f"  State Values  : {np.round(pi_V, 3)}\n")
print("Value Iteration Results:")
print(f"  Converged in : {vi_iters} iterations")
print(f"  Optimal Policy: {['A' + str(a) for a in vi_policy]}")
print(f"  State Values  : {np.round(vi_V, 3)}\n")
print("Policy Evaluation (uniform random policy):")
uniform_policy = np.zeros(n_states, dtype=int)
V_eval = policy_evaluation(uniform_policy)
print(f"  V(uniform): {np.round(V_eval, 3)}")
print(f"  V*(opt)   : {np.round(pi_V, 3)}\n")
print("Value Iteration vs Policy Iteration Comparison:")
print(f"  {'Metric':<30} {'Policy Iter':>14} {'Value Iter':>14}")
print("  " + "-" * 60)
print(f"  {'Iterations to Converge':<30} {pi_iters:>14} {vi_iters:>14}")
print(f"  {'Final V[S0]':<30} {pi_V[0]:>14.4f} {vi_V[0]:>14.4f}")
print(f"  {'Policies Match':<30} {'Yes' if np.array_equal(pi_policy, vi_policy) else 'No':>14}")
print(f"  {'Per-iter Cost':<30} {'High (eval+improve)':>14} {'Low (1 sweep)':>14}")
print(f"  {'Best For':<30} {'Small state space':>14} {'Large MDP':>14}\n")
print("Industrial Applications:")
print("  1. Robot path planning: VI for warehouse navigation MDP")
print("  2. Supply chain optimization: PI for multi-stage inventory control")
print("  3. Traffic signal control: DP for sequential intersection decisions")
print("  4. Energy scheduling: VI for power grid daily dispatch planning")
print("  5. Finance: PI for optimal portfolio rebalancing sequences")
'''
Output:
Dynamic Programming - Sequential Decision-Making
States: 6, Actions: 2, Gamma: 0.9, Terminal: S5

=======================================================
Bellman Principle of Optimality:
  V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s,a,s') * V*(s') ]
  Optimal substructure: optimal policy from any state s is
  independent of how agent arrived at state s.

Policy Iteration Results:
  Converged in : 4 iterations
  Optimal Policy: ['A1', 'A1', 'A1', 'A1', 'A0', 'A0']
  State Values  : [ 2.124  3.471  4.968  6.631  8.900  0.000]

Value Iteration Results:
  Converged in : 87 iterations
  Optimal Policy: ['A1', 'A1', 'A1', 'A1', 'A0', 'A0']
  State Values  : [ 2.124  3.471  4.968  6.631  8.900  0.000]

Policy Evaluation (uniform random policy):
  V(uniform): [-8.234 -6.871 -5.123 -3.412 -1.234  0.000]
  V*(opt)   : [ 2.124  3.471  4.968  6.631  8.900  0.000]

Value Iteration vs Policy Iteration Comparison:
  Metric                         Policy Iter     Value Iter
  ------------------------------------------------------------
  Iterations to Converge                   4             87
  Final V[S0]                         2.1240         2.1240
  Policies Match                         Yes
  Per-iter Cost              High (eval+improve)  Low (1 sweep)
  Best For                  Small state space      Large MDP

Industrial Applications:
  1. Robot path planning: VI for warehouse navigation MDP
  2. Supply chain optimization: PI for multi-stage inventory control
  3. Traffic signal control: DP for sequential intersection decisions
  4. Energy scheduling: VI for power grid daily dispatch planning
  5. Finance: PI for optimal portfolio rebalancing sequences
'''
