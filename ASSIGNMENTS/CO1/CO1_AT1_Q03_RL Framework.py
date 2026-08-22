# Question: To implement the basic Reinforcement Learning framework by designing an agent that interacts with an environment through states, actions, rewards, and policies using Python.
# Code:
import numpy as np
import gymnasium as gym
class RLAgent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = np.ones((n_states, n_actions)) / n_actions
    def select_action(self, state):
        return np.random.choice(self.n_actions, p=self.policy[state])
    def update_policy(self, state, action, reward):
        self.policy[state] = 0
        best = (action + 1) % self.n_actions if reward <= 0 else action
        self.policy[state][best] = 1.0
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
agent = RLAgent(n_states, n_actions)
print("RL Framework Initialized")
print(f"States: {n_states}, Actions: {n_actions}")
episode_rewards = []
for ep in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 20:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update_policy(state, action, reward)
        state = next_state
        total_reward += reward
        steps += 1
        if truncated:
            break
    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: Steps={steps}, Total Reward={total_reward}")
print(f"Average Reward: {np.mean(episode_rewards):.2f}")
env.close()
# Output:
# RL Framework Initialized
# States: 16, Actions: 4
# Episode 1: Steps=20, Total Reward=0.0
# Episode 2: Steps=20, Total Reward=0.0
# Episode 3: Steps=8, Total Reward=1.0
# Episode 4: Steps=20, Total Reward=0.0
# Episode 5: Steps=12, Total Reward=1.0
# Average Reward: 0.40
