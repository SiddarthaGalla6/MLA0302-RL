'''Question: To simulate an RL agent in a simple environment and visualize cumulative rewards, episode rewards, and learning performance using Matplotlib graphs.'''

# Code:
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
np.random.seed(42)
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.3
n_episodes = 200
episode_rewards = []
cumulative_rewards = []
total = 0
print("Reward Visualization - Q-Learning on FrozenLake")
print(f"Episodes: {n_episodes}, Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\n")
for ep in range(n_episodes):
    state, _ = env.reset()
    ep_reward = 0
    done = False
    steps = 0
    while not done and steps < 50:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state, reward, done, truncated, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        ep_reward += reward
        steps += 1
        if truncated:
            break
    total += ep_reward
    episode_rewards.append(ep_reward)
    cumulative_rewards.append(total)
    if (ep + 1) % 50 == 0:
        avg = np.mean(episode_rewards[max(0, ep-49):ep+1])
        print(f"Episode {ep+1:>3}: Cumulative={total:.1f}, Avg(last 50)={avg:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(episode_rewards, alpha=0.5, label="Episode Reward")
window = 20
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, n_episodes), moving_avg, color='red', label=f"Moving Avg ({window})")
axes[0].set_title("Episode Rewards")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[1].plot(cumulative_rewards, color='green')
axes[1].set_title("Cumulative Reward")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Cumulative Reward")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/reward_visualization.png", dpi=100)
plt.close()
print("\nPlot saved as reward_visualization.png")
print(f"Final Cumulative Reward: {cumulative_rewards[-1]:.1f}")
print(f"Success Rate: {sum(episode_rewards)/n_episodes*100:.1f}%")
env.close()

'''
Output:
# Reward Visualization - Q-Learning on FrozenLake
# Episodes: 200, Alpha: 0.1, Gamma: 0.9, Epsilon: 0.3
# Episode  50: Cumulative=4.0, Avg(last 50)=0.080
# Episode 100: Cumulative=14.0, Avg(last 50)=0.200
# Episode 150: Cumulative=28.0, Avg(last 50)=0.280
# Episode 200: Cumulative=45.0, Avg(last 50)=0.340
# Plot saved as reward_visualization.png
# Final Cumulative Reward: 45.0
# Success Rate: 22.5%
'''
