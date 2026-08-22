import numpy as np
np.random.seed(7)
n_arms, trials = 10, 500
true_rewards = np.random.normal(0, 1, n_arms)
def run_random():
    Q = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    cumulative = []
    total_reward = 0
    for t in range(trials):
        action = np.random.randint(n_arms)
        reward = np.random.normal(true_rewards[action], 1)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]
        total_reward += reward
        cumulative.append(total_reward)
    return cumulative, total_reward
def run_epsilon_greedy(epsilon):
    Q = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    cumulative = []
    total_reward = 0
    for t in range(trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q)
        reward = np.random.normal(true_rewards[action], 1)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]
        total_reward += reward
        cumulative.append(total_reward)
    return cumulative, total_reward
random_cum, random_total = run_random()
eg_cum, eg_total = run_epsilon_greedy(0.1)
print("Random vs Epsilon-Greedy Action Selection under Fixed Trial Budget")
print("Trials:", trials, "Arms:", n_arms)
print("Random Strategy Total Reward:", round(random_total, 2))
print("Epsilon-Greedy (eps=0.1) Total Reward:", round(eg_total, 2))
print("Random cumulative at trial 100:", round(random_cum[99], 2))
print("Epsilon-Greedy cumulative at trial 100:", round(eg_cum[99], 2))
print("Random cumulative at trial 500:", round(random_cum[-1], 2))
print("Epsilon-Greedy cumulative at trial 500:", round(eg_cum[-1], 2))
better = "Epsilon-Greedy" if eg_total > random_total else "Random"
print("Better performing strategy:", better)
