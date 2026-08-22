import numpy as np
import tensorflow as tf
np.random.seed(3)
tf.random.set_seed(3)
size = 5
n_states = size * size
n_actions = 4
restricted = [(1, 2), (2, 2), (3, 2)]
goal = (4, 4)
def state_id(r, c):
    return r * size + c
def is_restricted(r, c):
    return (r, c) in restricted
rewards = np.full((n_states,), -1.0)
rewards[state_id(*goal)] = 20.0
for r, c in restricted:
    rewards[state_id(r, c)] = -20.0
moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
V = tf.Variable(tf.zeros([n_states]), dtype=tf.float32)
gamma = 0.9
theta = 1e-3
def next_state(r, c, action):
    dr, dc = moves[action]
    nr, nc = min(max(r + dr, 0), size - 1), min(max(c + dc, 0), size - 1)
    if is_restricted(nr, nc):
        return r, c
    return nr, nc
for iteration in range(100):
    delta = 0.0
    new_V = V.numpy().copy()
    for r in range(size):
        for c in range(size):
            s = state_id(r, c)
            if (r, c) == goal:
                continue
            q_values = []
            for a in range(n_actions):
                nr, nc = next_state(r, c, a)
                ns = state_id(nr, nc)
                q_values.append(rewards[ns] + gamma * V.numpy()[ns])
            best_value = max(q_values)
            delta = max(delta, abs(best_value - new_V[s]))
            new_V[s] = best_value
    V.assign(new_V)
    if delta < theta:
        break
policy = np.zeros((size, size), dtype=int)
for r in range(size):
    for c in range(size):
        if (r, c) == goal:
            continue
        q_values = [rewards[state_id(*next_state(r, c, a))] + gamma * V.numpy()[state_id(*next_state(r, c, a))] for a in range(n_actions)]
        policy[r, c] = int(np.argmax(q_values))
print("Value Iteration on Constrained Grid (TensorFlow)")
print("Grid size:", size, "x", size)
print("Restricted states:", restricted)
print("Goal state:", goal)
print("Converged after iteration:", iteration)
print("Value Function:\n", V.numpy().reshape(size, size).round(2))
print("Optimal Policy (0=up 1=down 2=left 3=right):\n", policy)
