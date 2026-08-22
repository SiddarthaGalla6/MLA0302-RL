# Question: To install and configure Anaconda Navigator, Python, TensorFlow, Keras, Gymnasium (OpenAI Gym), NumPy, Matplotlib, 
#and other required libraries, and verify the environment by executing a simple Reinforcement Learning program.

Code:
import sys
import numpy
import matplotlib
import gymnasium
import tensorflow
import keras
print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Gymnasium:", gymnasium.__version__)
print("TensorFlow:", tensorflow.__version__)
print("Keras:", keras.__version__)
env = gymnasium.make("CartPole-v1")
state, _ = env.reset()
total_reward = 0
for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    if done or truncated:
        break
env.close()
print("Environment: CartPole-v1")
print("Initial State Shape:", state.shape)
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)
print("Total Reward (10 steps):", total_reward)
print("All libraries verified successfully.")


Output:
Python: 3.10.x (default, ...)
NumPy: 1.26.x
Matplotlib: 3.8.x
Gymnasium: 0.29.x
TensorFlow: 2.15.x
Keras: 2.15.x
Environment: CartPole-v1
Initial State Shape: (4,)
Action Space: Discrete(2)
Observation Space: Box([-4.8 ... 4.8], ...)
Total Reward (10 steps): 10.0
All libraries verified successfully.
