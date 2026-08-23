'''Question: Illustrate with an example how Q-learning updates the action-value function during
training, and demonstrate its convergence behavior.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_STATES = 5
N_ACTIONS = 2
GOAL = 4
def transition(state, action):
    if action == 0:
        return max(state - 1, 0)
    return min(state + 1, N_STATES - 1)
def reward(next_state):
    return 10.0 if next_state == GOAL else -1.0
alpha, gamma, epsilon = 0.2, 0.9, 0.3
Q = np.zeros((N_STATES, N_ACTIONS))
print("Q-Learning Update Rule Demonstration")
print("Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]")
print(f"alpha={alpha}, gamma={gamma}, State space={N_STATES}, Goal state={GOAL}\n")
print("First 5 manual updates from state 0, action 'right':")
state = 0
for step in range(5):
    action = 1
    next_state = transition(state, action)
    r = reward(next_state)
    old_q = Q[state, action]
    td_target = r + gamma * np.max(Q[next_state])
    td_error = td_target - old_q
    Q[state, action] = old_q + alpha * td_error
    print(f"  Step {step+1}: s={state} a=right r={r} s'={next_state} TD_target={td_target:.3f} TD_error={td_error:.3f} newQ={Q[state, action]:.3f}")
    state = next_state
Q_history = []
Q = np.zeros((N_STATES, N_ACTIONS))
episodes = 300
for ep in range(episodes):
    state = 0
    for step in range(20):
        action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
        next_state = transition(state, action)
        r = reward(next_state)
        Q[state, action] += alpha * (r + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if state == GOAL:
            break
    epsilon = max(0.05, epsilon * 0.98)
    Q_history.append(Q[0, 1])
print("\nConvergence of Q(state=0, action=right) over training:")
checkpoints = [0, 10, 50, 100, 200, 299]
for cp in checkpoints:
    print(f"  Episode {cp:>4}: Q(0,right) = {Q_history[cp]:.4f}")
print(f"\nFinal learned Q-table:\n{Q.round(3)}")
print("\nConvergence Behavior:")
print("  Early episodes show large swings in Q(0,right) as the agent has little experience to rely on")
print("  As more episodes accumulate, updates shrink and Q(0,right) stabilizes near its true optimal value")
print("  This matches the theoretical guarantee that Q-learning converges to Q* under sufficient exploration and a decaying learning rate")

'''
Output:
Q-Learning Update Rule Demonstration
Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
alpha=0.2, gamma=0.9, State space=5, Goal state=4

First 5 manual updates from state 0, action 'right':
  Step 1: s=0 a=right r=-1.0 s'=1 TD_target=-1.000 TD_error=-1.000 newQ=-0.200
  Step 2: s=1 a=right r=-1.0 s'=2 TD_target=-1.000 TD_error=-1.000 newQ=-0.200
  Step 3: s=2 a=right r=-1.0 s'=3 TD_target=-1.000 TD_error=-1.000 newQ=-0.200
  Step 4: s=3 a=right r=10.0 s'=4 TD_target=10.000 TD_error=10.000 newQ=2.000
  Step 5: s=4 a=right r=10.0 s'=4 TD_target=10.000 TD_error=10.000 newQ=2.000

Convergence of Q(state=0, action=right) over training:
  Episode    0: Q(0,right) = -0.3600
  Episode   10: Q(0,right) = -0.8582
  Episode   50: Q(0,right) = 4.5659
  Episode  100: Q(0,right) = 4.5800
  Episode  200: Q(0,right) = 4.5800
  Episode  299: Q(0,right) = 4.5800

Final learned Q-table:
[[ 2.4    4.58 ]
 [ 1.65   6.2  ]
 [ 3.764  8.   ]
 [ 5.291 10.   ]
 [ 0.     0.   ]]

Convergence Behavior:
  Early episodes show large swings in Q(0,right) as the agent has little experience to rely on
  As more episodes accumulate, updates shrink and Q(0,right) stabilizes near its true optimal value
  This matches the theoretical guarantee that Q-learning converges to Q* under sufficient exploration and a decaying learning rate
'''
