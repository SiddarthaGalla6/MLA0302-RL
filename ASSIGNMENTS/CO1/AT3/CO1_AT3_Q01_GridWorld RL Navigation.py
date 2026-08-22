import numpy as np
import gymnasium as gym
from gymnasium import spaces
np.random.seed(0)
class GridWorldEnv(gym.Env):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(size * size)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.goal = (4, 4)
        self.start = (0, 0)
    def reset(self, seed=None):
        self.agent = list(self.start)
        return self._state(), {}
    def _state(self):
        return self.agent[0] * self.size + self.agent[1]
    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        nr = min(max(self.agent[0] + dr, 0), self.size - 1)
        nc = min(max(self.agent[1] + dc, 0), self.size - 1)
        if (nr, nc) in self.obstacles:
            reward, done = -5, False
        elif (nr, nc) == self.goal:
            self.agent = [nr, nc]
            reward, done = 20, True
        else:
            self.agent = [nr, nc]
            reward, done = -1, False
        return self._state(), reward, done, False, {}
env = GridWorldEnv()
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 500
for ep in range(episodes):
    state, _ = env.reset()
    for step in range(50):
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state, reward, done, _, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if done:
            break
print("State Space Size:", env.observation_space.n)
print("Action Space Size:", env.action_space.n)
print("Obstacles:", env.obstacles)
print("Goal:", env.goal)
print("Trained Q-table shape:", Q.shape)
state, _ = env.reset()
path = [tuple(env.agent)]
for step in range(20):
    action = np.argmax(Q[state])
    state, reward, done, _, _ = env.step(action)
    path.append(tuple(env.agent))
    if done:
        break
print("Optimal Path Found:", path)
