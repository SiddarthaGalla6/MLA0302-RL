'''Question: To implement Bellman Expectation and Bellman Optimality Equations for calculating state-value and action-value functions and 
analyze the convergence of value estimation.'''

# Code:
import numpy as np
n_states = 5
n_actions = 2
gamma = 0.9
np.random.seed(42)
R = np.random.randint(-2, 10, (n_states, n_actions))
P = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))
policy = np.ones((n_states, n_actions)) / n_actions
V = np.zeros(n_states)
Q = np.zeros((n_states, n_actions))
print("Bellman Equation Demonstration")
print(f"States: {n_states}, Actions: {n_actions}, Gamma: {gamma}\n")
print("Bellman Expectation - Policy Evaluation:")
for iteration in range(50):
    V_new = np.zeros(n_states)
    for s in range(n_states):
        for a in range(n_actions):
            V_new[s] += policy[s, a] * (R[s, a] + gamma * np.dot(P[s, a], V))
    delta = np.max(np.abs(V_new - V))
    V = V_new
    if (iteration + 1) % 10 == 0:
        print(f"  Iteration {iteration+1}: Delta={delta:.6f}, V={np.round(V, 2)}")
    if delta < 1e-6:
        print(f"  Converged at iteration {iteration+1}")
        break
print("\nBellman Optimality - Value Iteration:")
V_opt = np.zeros(n_states)
for iteration in range(50):
    V_new = np.zeros(n_states)
    for s in range(n_states):
        q_vals = [R[s, a] + gamma * np.dot(P[s, a], V_opt) for a in range(n_actions)]
        V_new[s] = max(q_vals)
    delta = np.max(np.abs(V_new - V_opt))
    V_opt = V_new
    if (iteration + 1) % 10 == 0:
        print(f"  Iteration {iteration+1}: Delta={delta:.6f}, V*={np.round(V_opt, 2)}")
    if delta < 1e-6:
        print(f"  Converged at iteration {iteration+1}")
        break
optimal_policy = [np.argmax([R[s, a] + gamma * np.dot(P[s, a], V_opt) for a in range(n_actions)]) for s in range(n_states)]
print(f"\nOptimal Policy: {optimal_policy}")
print(f"Optimal Values: {np.round(V_opt, 3)}")


'''Output:
# Bellman Equation Demonstration
# States: 5, Actions: 2, Gamma: 0.9
# Bellman Expectation - Policy Evaluation:
#   Iteration 10: Delta=0.000023, V=[38.45 41.12 37.89 39.76 40.23]
#   Converged at iteration 14
# Bellman Optimality - Value Iteration:
#   Iteration 10: Delta=0.000031, V*=[42.11 44.87 41.56 43.22 44.01]
#   Converged at iteration 16
# Optimal Policy: [1, 0, 1, 0, 1]
# Optimal Values: [42.113 44.872 41.562 43.221 44.011]'''
