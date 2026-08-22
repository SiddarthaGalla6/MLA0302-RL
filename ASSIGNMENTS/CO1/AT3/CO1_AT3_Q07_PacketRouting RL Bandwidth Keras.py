import numpy as np
import tensorflow as tf
from tensorflow import keras
np.random.seed(6)
tf.random.set_seed(6)
N_ROUTES = 4
MAX_BANDWIDTH = 100.0
STATE_DIM = N_ROUTES + 1
model = keras.Sequential([
    keras.layers.Input(shape=(STATE_DIM,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(N_ROUTES, activation="linear")
])
optimizer = keras.optimizers.Adam(0.01)
route_capacity = np.array([30.0, 25.0, 20.0, 25.0], dtype=np.float32)
def get_state(usage, packet_size):
    return np.concatenate([usage / route_capacity, [packet_size / MAX_BANDWIDTH]]).astype(np.float32).reshape(1, -1)
def reward_function(route, packet_size, usage):
    if usage[route] + packet_size > route_capacity[route]:
        return -10.0
    congestion = usage[route] / route_capacity[route]
    return 5.0 - congestion * 5.0
def train_step(state, action, target_value):
    with tf.GradientTape() as tape:
        q_pred = model(state, training=True)
        target = tf.tensor_scatter_nd_update(q_pred[0], [[action]], [target_value])
        loss = tf.reduce_mean(tf.square(q_pred[0] - tf.stop_gradient(target)))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
epsilon, gamma, episodes = 0.3, 0.9, 60
for ep in range(episodes):
    usage = np.zeros(N_ROUTES, dtype=np.float32)
    total_reward = 0
    for step in range(15):
        packet_size = np.random.uniform(2, 8)
        state = get_state(usage, packet_size)
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ROUTES)
        else:
            action = int(np.argmax(model(state, training=False)[0].numpy()))
        reward = reward_function(action, packet_size, usage)
        if usage[action] + packet_size <= route_capacity[action]:
            usage[action] += packet_size
        next_state = get_state(usage, packet_size)
        next_q = model(next_state, training=False)[0].numpy()
        target_value = reward + gamma * np.max(next_q)
        train_step(state, action, target_value)
        total_reward += reward
    epsilon = max(0.05, epsilon * 0.95)
    usage = np.maximum(usage - 5.0, 0)
print("Packet Routing RL under Bandwidth Constraint (Keras)")
print("Number of routes:", N_ROUTES)
print("Route capacities (Mbps):", route_capacity)
print("Final episode total reward:", round(total_reward, 2))
test_state = get_state(np.array([10.0, 5.0, 18.0, 3.0], dtype=np.float32), 6.0)
q_values = model(test_state, training=False)[0].numpy()
print("Predicted Q-values for test state:", q_values.round(2))
print("Recommended route:", int(np.argmax(q_values)))
