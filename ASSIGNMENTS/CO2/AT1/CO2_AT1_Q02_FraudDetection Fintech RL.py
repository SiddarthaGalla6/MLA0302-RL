'''Question: Apply Reinforcement Learning concepts to a fraud detection system in fintech.
Define the state space, action space, and reward mechanism. Explain how
exploration strategies improve detection accuracy.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
AMOUNT_BINS = ['low', 'medium', 'high', 'very_high']
LOCATION_BINS = ['usual', 'unusual']
TIME_BINS = ['normal_hours', 'odd_hours']
FREQ_BINS = ['normal_freq', 'high_freq']
N_AMOUNT, N_LOCATION, N_TIME, N_FREQ = len(AMOUNT_BINS), len(LOCATION_BINS), len(TIME_BINS), len(FREQ_BINS)
N_STATES = N_AMOUNT * N_LOCATION * N_TIME * N_FREQ
ACTIONS = ['approve', 'flag_review', 'block']
N_ACTIONS = len(ACTIONS)
def encode_state(amount, location, time, freq):
    return amount * N_LOCATION * N_TIME * N_FREQ + location * N_TIME * N_FREQ + time * N_FREQ + freq
def generate_transaction():
    is_fraud = random.random() < 0.15
    if is_fraud:
        amount = random.choices(range(N_AMOUNT), weights=[0.1, 0.2, 0.3, 0.4])[0]
        location = random.choices(range(N_LOCATION), weights=[0.3, 0.7])[0]
        time = random.choices(range(N_TIME), weights=[0.3, 0.7])[0]
        freq = random.choices(range(N_FREQ), weights=[0.2, 0.8])[0]
    else:
        amount = random.choices(range(N_AMOUNT), weights=[0.5, 0.3, 0.15, 0.05])[0]
        location = random.choices(range(N_LOCATION), weights=[0.85, 0.15])[0]
        time = random.choices(range(N_TIME), weights=[0.8, 0.2])[0]
        freq = random.choices(range(N_FREQ), weights=[0.9, 0.1])[0]
    return amount, location, time, freq, is_fraud
def compute_fraud_reward(action, is_fraud):
    action_name = ACTIONS[action]
    if is_fraud:
        if action_name == 'block':
            return 10.0
        elif action_name == 'flag_review':
            return 4.0
        else:
            return -15.0
    else:
        if action_name == 'approve':
            return 3.0
        elif action_name == 'flag_review':
            return -2.0
        else:
            return -8.0
def run_fraud_rl(epsilon, n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma = 0.15, 0.9
    correct_blocks, false_blocks, missed_fraud, total_fraud = 0, 0, 0, 0
    for ep in range(n_episodes):
        amount, location, time, freq, is_fraud = generate_transaction()
        state = encode_state(amount, location, time, freq)
        if random.random() < epsilon:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = int(np.argmax(Q[state]))
        reward = compute_fraud_reward(action, is_fraud)
        next_amount, next_location, next_time, next_freq, _ = generate_transaction()
        next_state = encode_state(next_amount, next_location, next_time, next_freq)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        if is_fraud:
            total_fraud += 1
            if ACTIONS[action] == 'block':
                correct_blocks += 1
            elif ACTIONS[action] == 'approve':
                missed_fraud += 1
        else:
            if ACTIONS[action] == 'block':
                false_blocks += 1
    detection_rate = correct_blocks / total_fraud * 100 if total_fraud else 0
    return detection_rate, missed_fraud, false_blocks, total_fraud
print("Fraud Detection RL in Fintech Transactions")
print(f"State Space: amount x location x time x frequency = {N_STATES}")
print(f"Actions: {ACTIONS}")
print(f"{'Epsilon':>8} {'DetectRate%':>12} {'MissedFraud':>12} {'FalseBlocks':>12} {'TotalFraud':>11}")
print("-" * 60)
for eps in [0.05, 0.2, 0.4]:
    detection_rate, missed_fraud, false_blocks, total_fraud = run_fraud_rl(eps)
    print(f"{eps:>8} {detection_rate:>12.2f} {missed_fraud:>12} {false_blocks:>12} {total_fraud:>11}")
print("\nMDP Component Summary:")
print("  States : transaction amount x location x time x frequency pattern")
print("  Actions:", ACTIONS)
print("  Reward : +block correct fraud, +approve legit, -miss fraud, -false block")
print("\nExploration Strategy Impact:")
print("  Low epsilon converges fast but risks missing rare/novel fraud patterns")
print("  Higher epsilon explores more transaction patterns, improving long-run detection accuracy")
print("  Exploration is essential since fraud patterns evolve and static rules cannot adapt")

'''
Output:
Fraud Detection RL in Fintech Transactions
State Space: amount x location x time x frequency = 32
Actions: ['approve', 'flag_review', 'block']
 Epsilon  DetectRate%  MissedFraud  FalseBlocks  TotalFraud
------------------------------------------------------------
    0.05         1.30           23           11          77
     0.2        16.67           29           36          78
     0.4        34.18           32           58          79

MDP Component Summary:
  States : transaction amount x location x time x frequency pattern
  Actions: ['approve', 'flag_review', 'block']
  Reward : +block correct fraud, +approve legit, -miss fraud, -false block

Exploration Strategy Impact:
  Low epsilon converges fast but risks missing rare/novel fraud patterns
  Higher epsilon explores more transaction patterns, improving long-run detection accuracy
  Exploration is essential since fraud patterns evolve and static rules cannot adapt
'''
