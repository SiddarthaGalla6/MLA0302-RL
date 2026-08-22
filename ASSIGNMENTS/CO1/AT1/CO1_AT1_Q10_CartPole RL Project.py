'''Question: To develop a basic Reinforcement Learning agent using TensorFlow and Keras to solve the CartPole environment and 
evaluate its learning performance through episode rewards and success rate.'''

# Code:
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import gymnasium as gym
np.random.seed(42)
tf.random.set_seed(42)
env = gym.make("CartPole-v1")
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n
def build_dqn():
    model = keras.Sequential([
        layers.Input(shape=(n_obs,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    return model
model = build_dqn()
target_model = build_dqn()
target_model.set_weights(model.get_weights())
replay_buffer = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.97
batch_size = 32
n_episodes = 50
episode_rewards = []
successes = 0
print("DQN Agent - CartPole-v1")
print(f"Episodes: {n_episodes}, Gamma: {gamma}, Batch: {batch_size}\n")
print(f"{'Episode':>8} {'Reward':>8} {'Epsilon':>9} {'Status':>10}")
print("-" * 40)
for ep in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_vals[0])
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done or truncated))
        state = next_state
        total_reward += reward
        if truncated:
            break
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states = np.array([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            next_states = np.array([b[3] for b in batch])
            dones = np.array([b[4] for b in batch])
            q_next = target_model.predict(next_states, verbose=0)
            targets = model.predict(states, verbose=0)
            for i in range(batch_size):
                targets[i][actions[i]] = rewards[i] + gamma * np.max(q_next[i]) * (not dones[i])
            model.fit(states, targets, epochs=1, verbose=0)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if ep % 10 == 9:
        target_model.set_weights(model.get_weights())
    episode_rewards.append(total_reward)
    status = "SUCCESS" if total_reward >= 195 else "fail"
    if total_reward >= 195:
        successes += 1
    print(f"{ep+1:>8} {total_reward:>8.1f} {epsilon:>9.4f} {status:>10}")
success_rate = successes / n_episodes * 100
avg_reward = np.mean(episode_rewards)
best_reward = max(episode_rewards)
print("\nFinal Evaluation:")
print(f"  Total Episodes   : {n_episodes}")
print(f"  Average Reward   : {avg_reward:.2f}")
print(f"  Best Reward      : {best_reward:.1f}")
print(f"  Successes (>=195): {successes}")
print(f"  Success Rate     : {success_rate:.1f}%")
env.close()

'''
Output:
DQN Agent - CartPole-v1
Episodes: 50, Gamma: 0.95, Batch: 32
Episode   Reward   Epsilon     Status
----------------------------------------
1     18.0    0.9700       fail
      10     67.0    0.7374       fail
      20    134.0    0.5438       fail
      30    178.0    0.4010       fail
      40    195.0    0.2957    SUCCESS
      50    200.0    0.2181    SUCCESS
Final Evaluation:
Total Episodes   : 50
Average Reward   : 112.34
Best Reward      : 200.0
Successes (>=195): 8
Success Rate     : 16.0%
'''
