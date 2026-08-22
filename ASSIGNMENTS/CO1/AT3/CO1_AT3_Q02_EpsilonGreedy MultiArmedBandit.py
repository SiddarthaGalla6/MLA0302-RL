import numpy as np
np.random.seed(1)
n_arms, trials = 10, 500
true_rewards = np.random.normal(0, 1, n_arms)
def run_bandit(epsilon):
    Q = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    total_reward = 0
    explore_count = 0
    rewards_history = []
    for t in range(trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
            explore_count += 1
        else:
            action = np.argmax(Q)
        reward = np.random.normal(true_rewards[action], 1)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]
        total_reward += reward
        rewards_history.append(total_reward)
    exploit_count = trials - explore_count
    return total_reward, explore_count, exploit_count, Q, rewards_history
print("Multi-Armed Bandit with Epsilon-Greedy Strategy")
print("Number of Arms:", n_arms, "Trials:", trials)
print(f"{'Epsilon':>8} {'TotalReward':>12} {'Explore':>8} {'Exploit':>8} {'BestArm':>8}")
for eps in [0.1, 0.3, 0.5]:
    total_reward, explore_count, exploit_count, Q, history = run_bandit(eps)
    best_arm = np.argmax(Q)
    print(f"{eps:>8} {total_reward:>12.2f} {explore_count:>8} {exploit_count:>8} {best_arm:>8}")
print("True optimal arm:", np.argmax(true_rewards))
print("Constraint: fixed budget of", trials, "trials satisfied for all epsilon values")
