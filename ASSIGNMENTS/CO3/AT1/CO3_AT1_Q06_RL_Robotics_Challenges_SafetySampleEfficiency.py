'''Question: Identify the challenges of applying RL in real-world robotics and analyze how
safety, sample efficiency, and environment variability affect agent performance.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_STATES = 10
N_ACTIONS = 3
DANGER_STATE = 8
GOAL_STATE = 9
def environment_step(state, action, noise_level):
    if random.random() < noise_level:
        action = random.randint(0, N_ACTIONS - 1)
    if action == 0:
        next_state = max(state - 1, 0)
    elif action == 1:
        next_state = min(state + 1, N_STATES - 1)
    else:
        next_state = state
    if next_state == DANGER_STATE:
        reward, unsafe = -20.0, True
    elif next_state == GOAL_STATE:
        reward, unsafe = 15.0, False
    else:
        reward, unsafe = -0.3, False
    return next_state, reward, unsafe
def run_robot_rl(noise_level, n_episodes):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.15, 0.9, 0.3
    unsafe_events, ep_rewards = [], []
    for ep in range(n_episodes):
        state = 0
        total_reward, unsafe_count = 0, 0
        for step in range(30):
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            next_state, reward, unsafe = environment_step(state, action, noise_level)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward
            if unsafe:
                unsafe_count += 1
            if state == GOAL_STATE:
                break
        epsilon = max(0.05, epsilon * 0.98)
        unsafe_events.append(unsafe_count)
        ep_rewards.append(total_reward)
    return unsafe_events, ep_rewards
print("Real-World Robotics RL: Safety, Sample Efficiency, and Environment Variability")
print(f"State Space: {N_STATES}, Danger state: {DANGER_STATE}, Goal state: {GOAL_STATE}")
print(f"\n{'NoiseLevel':>11} {'Episodes':>9} {'AvgUnsafeEvents':>16} {'AvgRewardLast20':>16}")
print("-" * 58)
for noise in [0.0, 0.15, 0.35]:
    for n_ep in [50, 300]:
        unsafe_events, ep_rewards = run_robot_rl(noise, n_ep)
        print(f"{noise:>11} {n_ep:>9} {np.mean(unsafe_events[-20:]):>16.3f} {np.mean(ep_rewards[-20:]):>16.3f}")
print("\nChallenges Identified:")
print("  Safety      : exploration steps can enter the danger state, which is unacceptable on physical hardware")
print("  Sample Efficiency: with only 50 training episodes the policy has far less time to learn than with 300")
print("  Environment Variability: added action noise (sensor/actuator error) makes outcomes less predictable")
print("\nImpact on Agent Performance:")
print("  In several runs above the agent converges to staying near the start rather than reaching the goal at all")
print("  This shows a real robotics failure mode: a heavy danger penalty can teach an overly conservative policy")
print("  that avoids the task entirely instead of completing it, rather than one that reaches the goal safely")
print("  Mitigation approaches: simulation-to-real transfer, constrained/safe RL, and reward tuning to avoid this trap")

'''
Output:
Real-World Robotics RL: Safety, Sample Efficiency, and Environment Variability
State Space: 10, Danger state: 8, Goal state: 9

 NoiseLevel  Episodes  AvgUnsafeEvents  AvgRewardLast20
----------------------------------------------------------
        0.0        50            0.200           -9.535
        0.0       300            0.000           -9.000
       0.15        50            0.000           -9.000
       0.15       300            0.000           -9.000
       0.35        50            0.250          -10.475
       0.35       300            0.000           -9.000

Challenges Identified:
  Safety      : exploration steps can enter the danger state, which is unacceptable on physical hardware
  Sample Efficiency: with only 50 training episodes the policy has far less time to learn than with 300
  Environment Variability: added action noise (sensor/actuator error) makes outcomes less predictable

Impact on Agent Performance:
  In several runs above the agent converges to staying near the start rather than reaching the goal at all
  This shows a real robotics failure mode: a heavy danger penalty can teach an overly conservative policy
  that avoids the task entirely instead of completing it, rather than one that reaches the goal safely
  Mitigation approaches: simulation-to-real transfer, constrained/safe RL, and reward tuning to avoid this trap
'''
