'''Question: Evaluate the effectiveness of Reinforcement Learning in predictive maintenance
compared to traditional rule-based approaches. Discuss risks and reliability concerns
in industrial environments.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
WEAR_BINS = 10
VIBRATION_BINS = 3
N_STATES = WEAR_BINS * VIBRATION_BINS
ACTIONS = ['continue_operation', 'schedule_maintenance', 'stop_immediately']
N_ACTIONS = len(ACTIONS)
def encode_state(wear, vibration):
    return wear * VIBRATION_BINS + vibration
def simulate_machine_step(wear, action):
    if action == 0:
        wear = min(wear + random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0], WEAR_BINS - 1)
    elif action == 1:
        wear = max(wear - 4, 0)
    else:
        wear = 0
    failure_prob = (wear / WEAR_BINS) ** 2
    failed = random.random() < failure_prob
    vibration = min(int(wear / 3.5), VIBRATION_BINS - 1)
    return wear, vibration, failed
def compute_maintenance_reward(action, failed, wear):
    action_name = ACTIONS[action]
    if failed:
        return -50.0
    if action_name == 'continue_operation':
        return 5.0 - 0.3 * wear
    elif action_name == 'schedule_maintenance':
        return -3.0
    else:
        return -8.0
def run_rl_policy(n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    failures, uptime, episodes_cost = 0, 0, []
    for ep in range(n_episodes):
        wear, vibration = random.randint(0, 2), 0
        total_cost = 0
        for step in range(30):
            state = encode_state(wear, vibration)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            new_wear, new_vibration, failed = simulate_machine_step(wear, action)
            reward = compute_maintenance_reward(action, failed, wear)
            next_state = encode_state(new_wear, new_vibration)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            wear, vibration = new_wear, new_vibration
            total_cost += -reward
            if failed:
                failures += 1
            else:
                uptime += 1
        epsilon = max(0.05, epsilon * 0.99)
        episodes_cost.append(total_cost)
    return failures, uptime, np.mean(episodes_cost[-50:])
def run_rule_based(n_episodes=400):
    failures, uptime, episodes_cost = 0, 0, []
    for ep in range(n_episodes):
        wear = random.randint(0, 2)
        total_cost = 0
        for step in range(30):
            action = 1 if wear >= 6 else 0
            new_wear, new_vibration, failed = simulate_machine_step(wear, action)
            reward = compute_maintenance_reward(action, failed, wear)
            wear = new_wear
            total_cost += -reward
            if failed:
                failures += 1
            else:
                uptime += 1
        episodes_cost.append(total_cost)
    return failures, uptime, np.mean(episodes_cost[-50:])
print("Predictive Maintenance: RL Policy vs Rule-Based Threshold Policy")
print(f"State Space: wear_level x vibration_level = {N_STATES}")
print(f"Actions: {ACTIONS}")
rl_failures, rl_uptime, rl_cost = run_rl_policy()
rule_failures, rule_uptime, rule_cost = run_rule_based()
print(f"\n{'Approach':>15} {'Failures':>10} {'Uptime':>10} {'AvgCost':>10}")
print("-" * 48)
print(f"{'RL Policy':>15} {rl_failures:>10} {rl_uptime:>10} {rl_cost:>10.2f}")
print(f"{'Rule-Based':>15} {rule_failures:>10} {rule_uptime:>10} {rule_cost:>10.2f}")
print("\nEffectiveness Summary:")
print("  RL adapts maintenance timing to observed wear and vibration patterns, not a fixed threshold")
print("  Rule-based policy is simple and predictable but reacts only after crossing a static threshold")
print("\nRisks and Reliability Concerns in Industrial Deployment:")
print("  RL exploration during training can under-maintain equipment and cause real failures")
print("  Black-box Q-values are harder to audit than transparent rule thresholds for safety certification")
print("  Mitigation: train in simulation first, deploy with conservative action bounds and human override")

'''
Output:
Predictive Maintenance: RL Policy vs Rule-Based Threshold Policy
State Space: wear_level x vibration_level = 30
Actions: ['continue_operation', 'schedule_maintenance', 'stop_immediately']

       Approach   Failures     Uptime    AvgCost
------------------------------------------------
      RL Policy        105      11895     -40.34
     Rule-Based       1844      10156     144.99

Effectiveness Summary:
  RL adapts maintenance timing to observed wear and vibration patterns, not a fixed threshold
  Rule-based policy is simple and predictable but reacts only after crossing a static threshold

Risks and Reliability Concerns in Industrial Deployment:
  RL exploration during training can under-maintain equipment and cause real failures
  Black-box Q-values are harder to audit than transparent rule thresholds for safety certification
  Mitigation: train in simulation first, deploy with conservative action bounds and human override
'''
