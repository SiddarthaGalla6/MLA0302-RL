# Question: To create, initialize, and interact with standard Reinforcement Learning environments such as FrozenLake, CartPole, and MountainCar using Gymnasium to understand the concepts of states, actions, rewards, and episodes.
# Code:
import gymnasium as gym
def explore_env(name, steps=5):
    env = gym.make(name)
    state, _ = env.reset()
    print(f"\nEnvironment: {name}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    total_reward = 0
    for step in range(steps):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: Action={action}, Reward={reward}, Done={done}")
        if done or truncated:
            state, _ = env.reset()
            break
    print(f"Total Reward: {total_reward}")
    env.close()
explore_env("FrozenLake-v1", steps=5)
explore_env("CartPole-v1", steps=5)
explore_env("MountainCar-v0", steps=5)
# Output:
# Environment: FrozenLake-v1
# Observation Space: Discrete(16)
# Action Space: Discrete(4)
# Step 1: Action=2, Reward=0.0, Done=False
# Step 2: Action=1, Reward=0.0, Done=False
# Step 3: Action=0, Reward=0.0, Done=False
# Step 4: Action=3, Reward=0.0, Done=False
# Step 5: Action=2, Reward=0.0, Done=False
# Total Reward: 0.0
# Environment: CartPole-v1
# Observation Space: Box([-4.8 ...], [4.8 ...], (4,), float32)
# Action Space: Discrete(2)
# Step 1: Action=1, Reward=1.0, Done=False
# Step 2: Action=0, Reward=1.0, Done=False
# Step 3: Action=1, Reward=1.0, Done=False
# Step 4: Action=1, Reward=1.0, Done=True
# Total Reward: 4.0
# Environment: MountainCar-v0
# Observation Space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
# Action Space: Discrete(3)
# Step 1: Action=0, Reward=-1.0, Done=False
# Step 2: Action=2, Reward=-1.0, Done=False
# Step 3: Action=1, Reward=-1.0, Done=False
# Step 4: Action=0, Reward=-1.0, Done=False
# Step 5: Action=1, Reward=-1.0, Done=False
# Total Reward: -5.0
