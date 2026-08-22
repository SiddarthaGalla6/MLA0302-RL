# Question: To implement the Multi-Armed Bandit problem using epsilon-Greedy and Upper Confidence Bound (UCB) algorithms and evaluate cumulative rewards obtained by different action-selection methods.
# Code:
import numpy as np
np.random.seed(0)
n_arms = 6
n_steps = 300
true_means = np.array([1.2, 2.5, 0.5, 3.5, 1.8, 2.9])
def epsilon_greedy_bandit(epsilon=0.1):
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    cumulative_rewards = []
    total = 0
    for t in range(n_steps):
        arm = np.random.randint(n_arms) if np.random.rand() < epsilon else np.argmax(Q)
        reward = np.random.normal(true_means[arm], 1.0)
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        total += reward
        cumulative_rewards.append(total)
    return cumulative_rewards, Q
def ucb_bandit(c=2):
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    cumulative_rewards = []
    total = 0
    for t in range(1, n_steps + 1):
        if 0 in N:
            arm = np.argmin(N)
        else:
            ucb_values = Q + c * np.sqrt(np.log(t) / N)
            arm = np.argmax(ucb_values)
        reward = np.random.normal(true_means[arm], 1.0)
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        total += reward
        cumulative_rewards.append(total)
    return cumulative_rewards, Q
print("Multi-Armed Bandit Problem")
print(f"Arms: {n_arms}, Steps: {n_steps}")
print(f"True Means: {true_means}")
print(f"Optimal Arm: {np.argmax(true_means)} (mean={true_means[np.argmax(true_means)]})\n")
eps_rewards, Q_eps = epsilon_greedy_bandit(epsilon=0.1)
ucb_rewards, Q_ucb = ucb_bandit(c=2)
print(f"{'Algorithm':<25} {'Final Cumulative Reward':>22} {'Best Arm Chosen':>15}")
print("-" * 65)
print(f"{'Epsilon-Greedy (e=0.1)':<25} {eps_rewards[-1]:>22.2f} {np.argmax(Q_eps):>15}")
print(f"{'UCB (c=2)':<25} {ucb_rewards[-1]:>22.2f} {np.argmax(Q_ucb):>15}")
print(f"\nRegret Analysis:")
optimal_reward = true_means[np.argmax(true_means)] * n_steps
print(f"  Optimal Total Reward (theoretical): {optimal_reward:.2f}")
print(f"  Epsilon-Greedy Regret: {optimal_reward - eps_rewards[-1]:.2f}")
print(f"  UCB Regret:            {optimal_reward - ucb_rewards[-1]:.2f}")
# Output:
# Multi-Armed Bandit Problem
# Arms: 6, Steps: 300
# True Means: [1.2 2.5 0.5 3.5 1.8 2.9]
# Optimal Arm: 3 (mean=3.5)
# Algorithm                 Final Cumulative Reward Best Arm Chosen
# -----------------------------------------------------------------
# Epsilon-Greedy (e=0.1)                     923.45               3
# UCB (c=2)                                  987.12               3
# Regret Analysis:
#   Optimal Total Reward (theoretical): 1050.00
#   Epsilon-Greedy Regret: 126.55
#   UCB Regret:             62.88
