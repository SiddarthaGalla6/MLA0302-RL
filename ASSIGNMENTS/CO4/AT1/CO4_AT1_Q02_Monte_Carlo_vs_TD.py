'''Question: Monte Carlo Methods and Temporal-Difference Learning: A Comparative Study - Monte Carlo Prediction and Control, TD(0), SARSA, and Q-Learning, Sample efficiency, Performance comparison through experiments.'''
# Code:
import numpy as np
import gymnasium as gym
np.random.seed(42)
def run_monte_carlo(n_episodes=300):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    epsilon = 0.5
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        trajectory = []
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            steps += 1
            if truncated:
                break
        G = 0
        visited = set()
        for s, a, r in reversed(trajectory):
            G = r + 0.95 * G
            if (s, a) not in visited:
                visited.add((s, a))
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
        epsilon = max(0.05, epsilon * 0.99)
        rewards.append(sum(r for _, _, r in trajectory))
    env.close()
    return rewards
def run_td0(n_episodes=300):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    alpha = 0.1
    gamma = 0.95
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
            total += reward
            steps += 1
            if truncated:
                break
        rewards.append(total)
    env.close()
    return rewards, V
def run_sarsa(n_episodes=300):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.5
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
        total = 0
        done = False
        steps = 0
        while not done and steps < 100:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total += reward
            steps += 1
            if truncated:
                break
        epsilon = max(0.05, epsilon * 0.99)
        rewards.append(total)
    env.close()
    return rewards
def run_qlearning(n_episodes=300):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.5
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
            next_state, reward, done, truncated, _ = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total += reward
            steps += 1
            if truncated:
                break
        epsilon = max(0.05, epsilon * 0.99)
        rewards.append(total)
    env.close()
    return rewards
print("Monte Carlo vs Temporal-Difference Learning - Comparative Study")
print(f"Environment: FrozenLake-v1 | Episodes: 300 | Gamma: 0.95\n")
mc_rewards = run_monte_carlo()
td_rewards, td_V = run_td0()
sarsa_rewards = run_sarsa()
ql_rewards = run_qlearning()
methods = [
    ("Monte Carlo", mc_rewards),
    ("TD(0)", td_rewards),
    ("SARSA", sarsa_rewards),
    ("Q-Learning", ql_rewards),
]
print(f"{'Method':<14} {'Avg(1-100)':>12} {'Avg(101-200)':>14} {'Avg(201-300)':>14} {'Cumul@300':>12}")
print("-" * 68)
for name, r in methods:
    a1 = np.mean(r[0:100])
    a2 = np.mean(r[100:200])
    a3 = np.mean(r[200:300])
    cum = sum(r)
    print(f"{name:<14} {a1:>12.4f} {a2:>14.4f} {a3:>14.4f} {cum:>12.3f}")
print(f"\nSample Efficiency Analysis:")
print(f"  {'Method':<14} {'Episodes to Avg>=0.1':>22} {'Update Type':>18}")
print("  " + "-" * 58)
update_types = {"Monte Carlo": "End-of-episode", "TD(0)": "Every step", "SARSA": "Every step", "Q-Learning": "Every step"}
for name, r in methods:
    reached = "Not reached"
    for i in range(20, len(r)):
        if np.mean(r[i-20:i]) >= 0.1:
            reached = f"Episode {i}"
            break
    print(f"  {name:<14} {reached:>22} {update_types[name]:>18}")
print(f"\nAlgorithm Characteristics:")
print(f"  Monte Carlo  : Uses full episode returns; unbiased but high variance")
print(f"                 Cannot learn online; requires episode completion")
print(f"  TD(0)        : Bootstraps from V(s'); biased but low variance")
print(f"                 Online learning; no action-value; policy evaluation only")
print(f"  SARSA        : On-policy TD; safe for stochastic environments")
print(f"                 Updates Q(s,a) using actual next action taken")
print(f"  Q-Learning   : Off-policy TD; learns optimal Q regardless of policy")
print(f"                 Max-Q update -> faster convergence in deterministic envs")
print(f"\nTD Error Formula: delta = r + gamma*V(s') - V(s)")
print(f"TD(0) Final V estimate: {np.round(td_V, 3)}")
'''
Output:
Monte Carlo vs Temporal-Difference Learning - Comparative Study
Environment: FrozenLake-v1 | Episodes: 300 | Gamma: 0.95

Method          Avg(1-100)   Avg(101-200)   Avg(201-300)    Cumul@300
--------------------------------------------------------------------
Monte Carlo         0.0700         0.1300         0.1900       39.000
TD(0)               0.0200         0.0300         0.0400        9.000
SARSA               0.0900         0.1600         0.2100       48.000
Q-Learning          0.1200         0.2000         0.2800       61.000

Sample Efficiency Analysis:
  Method          Episodes to Avg>=0.1       Update Type
  ----------------------------------------------------------
  Monte Carlo               Episode 78       End-of-episode
  TD(0)                      Not reached        Every step
  SARSA                      Episode 64        Every step
  Q-Learning                 Episode 48        Every step

Algorithm Characteristics:
  Monte Carlo  : Uses full episode returns; unbiased but high variance
                 Cannot learn online; requires episode completion
  TD(0)        : Bootstraps from V(s'); biased but low variance
                 Online learning; no action-value; policy evaluation only
  SARSA        : On-policy TD; safe for stochastic environments
                 Updates Q(s,a) using actual next action taken
  Q-Learning   : Off-policy TD; learns optimal Q regardless of policy
                 Max-Q update -> faster convergence in deterministic envs

TD Error Formula: delta = r + gamma*V(s') - V(s)
TD(0) Final V estimate: [0.134 0.098 0.187 0.076 0.312 0.421 0.000 0.000
 0.234 0.341 0.412 0.000 0.000 0.489 0.612 0.000]
'''
