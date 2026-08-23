'''Question: A healthcare monitoring system uses Reinforcement Learning to adjust medication
dosages. Apply RL concepts to define states, actions, and rewards, and explain how
learning improves outcomes.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
VITAL_BINS = 5
TREND_BINS = 3
N_STATES = VITAL_BINS * TREND_BINS
ACTIONS = ['increase_dose', 'decrease_dose', 'maintain_dose', 'alert_clinician']
N_ACTIONS = len(ACTIONS)
TARGET_LEVEL = 2
def encode_state(vital, trend):
    return vital * TREND_BINS + trend
def simulate_patient_step(vital, action):
    action_name = ACTIONS[action]
    if action_name == 'increase_dose':
        vital = min(vital + 1, VITAL_BINS - 1)
    elif action_name == 'decrease_dose':
        vital = max(vital - 1, 0)
    elif action_name == 'alert_clinician':
        vital = vital
    natural_drift = random.choices([-1, 0, 1], weights=[0.25, 0.5, 0.25])[0]
    vital = min(max(vital + natural_drift, 0), VITAL_BINS - 1)
    trend = 1 if vital > TARGET_LEVEL else (0 if vital == TARGET_LEVEL else 2)
    return vital, trend
def compute_health_reward(action, vital):
    action_name = ACTIONS[action]
    deviation = abs(vital - TARGET_LEVEL)
    stability_score = 5.0 - deviation * 2.0
    if vital in [0, VITAL_BINS - 1]:
        critical_penalty = -10.0
    else:
        critical_penalty = 0.0
    action_cost = -0.5 if action_name in ['increase_dose', 'decrease_dose'] else 0.0
    if action_name == 'alert_clinician' and deviation >= 2:
        alert_bonus = 3.0
    elif action_name == 'alert_clinician':
        alert_bonus = -1.0
    else:
        alert_bonus = 0.0
    return stability_score + critical_penalty + action_cost + alert_bonus
def run_healthcare_rl(n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_rewards, critical_events = [], []
    for ep in range(n_episodes):
        vital = random.randint(1, 3)
        trend = 0
        total_reward, critical_count = 0, 0
        for step in range(30):
            state = encode_state(vital, trend)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            reward = compute_health_reward(action, vital)
            new_vital, new_trend = simulate_patient_step(vital, action)
            if new_vital in [0, VITAL_BINS - 1]:
                critical_count += 1
            next_state = encode_state(new_vital, new_trend)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            vital, trend = new_vital, new_trend
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.99)
        ep_rewards.append(total_reward)
        critical_events.append(critical_count)
    return ep_rewards, critical_events
print("Healthcare Monitoring RL for Medication Dosage Adjustment")
print(f"State Space: vital_level x trend = {N_STATES}")
print(f"Actions: {ACTIONS}")
rewards, critical_events = run_healthcare_rl()
print(f"\nAvg reward first 50 episodes:          {np.mean(rewards[:50]):.3f}")
print(f"Avg reward last 50 episodes:           {np.mean(rewards[-50:]):.3f}")
print(f"Avg critical events first 50 episodes: {np.mean(critical_events[:50]):.3f}")
print(f"Avg critical events last 50 episodes:  {np.mean(critical_events[-50:]):.3f}")
print("\nRL Concept Mapping:")
print("  States : patient vital sign level x recent trend direction")
print("  Actions:", ACTIONS)
print("  Reward : +stability near target vital - critical range penalty - dosing action cost + alert bonus")
print("\nLearning Improving Outcomes:")
print("  Repeated interaction lets Q-values learn which dosage adjustments keep vitals near the safe target")
print("  Falling critical-event counts across episodes indicate the policy is stabilizing patients faster")

'''
Output:
Healthcare Monitoring RL for Medication Dosage Adjustment
State Space: vital_level x trend = 15
Actions: ['increase_dose', 'decrease_dose', 'maintain_dose', 'alert_clinician']

Avg reward first 50 episodes:          56.610
Avg reward last 50 episodes:           104.420
Avg critical events first 50 episodes: 4.280
Avg critical events last 50 episodes:  0.460

RL Concept Mapping:
  States : patient vital sign level x recent trend direction
  Actions: ['increase_dose', 'decrease_dose', 'maintain_dose', 'alert_clinician']
  Reward : +stability near target vital - critical range penalty - dosing action cost + alert bonus

Learning Improving Outcomes:
  Repeated interaction lets Q-values learn which dosage adjustments keep vitals near the safe target
  Falling critical-event counts across episodes indicate the policy is stabilizing patients faster
'''
