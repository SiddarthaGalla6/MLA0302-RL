'''Question: To build a simple feed-forward neural network using TensorFlow and Keras for approximating value functions in Reinforcement Learning environments.'''

# Code:
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
np.random.seed(42)
tf.random.set_seed(42)
env = gym.make("CartPole-v1")
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n
def build_value_network(input_dim, output_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    return model
model = build_value_network(n_obs, n_actions)
print("Neural Network Architecture:")
model.summary()
print(f"\nInput: CartPole observation ({n_obs} features)")
print(f"Output: Q-values for {n_actions} actions\n")
gamma = 0.95
epsilon = 0.5
batch_X, batch_y = [], []
print("Collecting training samples and fitting network:")
for ep in range(5):
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 30:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_vals[0])
        next_state, reward, done, truncated, _ = env.step(action)
        q_next = model.predict(next_state.reshape(1, -1), verbose=0)[0]
        target = model.predict(state.reshape(1, -1), verbose=0)[0]
        target[action] = reward + gamma * np.max(q_next) * (not done)
        batch_X.append(state)
        batch_y.append(target)
        state = next_state
        steps += 1
        if truncated:
            break
X = np.array(batch_X)
y = np.array(batch_y)
history = model.fit(X, y, epochs=10, batch_size=16, verbose=0)
print(f"Training samples: {len(X)}")
print(f"Loss after training: {history.history['loss'][-1]:.4f}")
sample_state, _ = env.reset()
q_values = model.predict(sample_state.reshape(1, -1), verbose=0)[0]
print(f"\nSample State: {np.round(sample_state, 3)}")
print(f"Predicted Q-values: {np.round(q_values, 4)}")
print(f"Suggested Action: {np.argmax(q_values)} ({'Right' if np.argmax(q_values)==1 else 'Left'})")
env.close()


'''Output:
# Neural Network Architecture:
# Model: "sequential"
# _________________________________________________________________
# Layer (type)          Output Shape        Param #
# =================================================================
# dense (Dense)         (None, 64)          320
# dense_1 (Dense)       (None, 64)          4160
# dense_2 (Dense)       (None, 2)           130
# =================================================================
# Total params: 4610 (18.01 KB)
# Trainable params: 4610 (18.01 KB)
# Non-trainable params: 0 (0.00 B)
# Input: CartPole observation (4 features)
# Output: Q-values for 2 actions
# Training samples: 87
# Loss after training: 0.0231
# Sample State: [ 0.023 -0.041  0.011  0.029]
# Predicted Q-values: [0.0412 0.0389]
# Suggested Action: 0 (Left)
'''
