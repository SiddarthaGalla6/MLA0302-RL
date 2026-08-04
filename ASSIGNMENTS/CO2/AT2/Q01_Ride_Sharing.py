"""
Ride Sharing using Reinforcement Learning
Simple Q-Learning Example
"""
import numpy as np
import random
states = ['Driver Available','Passenger Waiting']
actions = ['Assign Driver','Wait']
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 200
Q = np.zeros((len(states), len(actions)))
for episode in range(episodes):
    state = random.randint(0, len(states)-1)
    while True:
        if random.random() < epsilon:
            action = random.randint(0, len(actions)-1)
        else:
            action = np.argmax(Q[state])
        reward = 10 if action==0 else -5
        next_state = random.randint(0, len(states)-1)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state
        break
print("\nQ-Table")
print(Q)
print("\nLearned Policy")
for i, s in enumerate(states):
    print(f"{s} --> {actions[np.argmax(Q[i])]}")
