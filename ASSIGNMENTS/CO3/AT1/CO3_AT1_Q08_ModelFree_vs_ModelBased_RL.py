'''Question: Evaluate the effectiveness of model-free RL compared to model-based RL in
dynamic environments, and differentiate their strengths and limitations.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_STATES = 6
N_ACTIONS = 2
GOAL = 5
def true_transition(state, action, hazard_state):
    if action == 0:
        next_state = max(state - 1, 0)
    else:
        next_state = min(state + 1, N_STATES - 1)
    if next_state == hazard_state and random.random() < 0.5:
        next_state = state
    return next_state
def reward_fn(next_state):
    return 15.0 if next_state == GOAL else -0.5
def run_model_free(hazard_state, n_episodes):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.15, 0.9, 0.3
    for ep in range(n_episodes):
        state = 0
        for step in range(20):
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            next_state = true_transition(state, action, hazard_state)
            reward = reward_fn(next_state)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if state == GOAL:
                break
        epsilon = max(0.05, epsilon * 0.98)
    return Q
def run_model_based(hazard_state, n_episodes):
    model = {}
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon, planning_steps = 0.15, 0.9, 0.3, 5
    for ep in range(n_episodes):
        state = 0
        for step in range(20):
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            next_state = true_transition(state, action, hazard_state)
            reward = reward_fn(next_state)
            model[(state, action)] = (next_state, reward)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            for _ in range(planning_steps):
                if not model:
                    break
                s_p, a_p = random.choice(list(model.keys()))
                s_next_p, r_p = model[(s_p, a_p)]
                Q[s_p, a_p] += alpha * (r_p + gamma * np.max(Q[s_next_p]) - Q[s_p, a_p])
            state = next_state
            if state == GOAL:
                break
        epsilon = max(0.05, epsilon * 0.98)
    return Q
def evaluate_policy(Q, hazard_state, trials=100):
    successes, total_steps = 0, 0
    for t in range(trials):
        state = 0
        for step in range(20):
            action = int(np.argmax(Q[state]))
            state = true_transition(state, action, hazard_state)
            if state == GOAL:
                successes += 1
                total_steps += step + 1
                break
    return successes / trials * 100, total_steps / max(successes, 1)
print("Model-Free vs Model-Based RL in a Dynamic Environment")
print(f"State Space: {N_STATES}, Goal: {GOAL}")
print(f"\n{'EnvChange':>12} {'Episodes':>9} {'Approach':>13} {'SuccessRate%':>13} {'AvgStepsToGoal':>15}")
print("-" * 70)
for hazard, n_ep in [(3, 4), (2, 4)]:
    Q_free = run_model_free(hazard, n_ep)
    Q_based = run_model_based(hazard, n_ep)
    succ_free, steps_free = evaluate_policy(Q_free, hazard)
    succ_based, steps_based = evaluate_policy(Q_based, hazard)
    print(f"{'hazard=' + str(hazard):>12} {n_ep:>9} {'model-free':>13} {succ_free:>13.1f} {steps_free:>15.2f}")
    print(f"{'hazard=' + str(hazard):>12} {n_ep:>9} {'model-based':>13} {succ_based:>13.1f} {steps_based:>15.2f}")
print("\nEvaluation Summary:")
print("  Model-based RL reuses a learned transition model for extra planning updates, often learning faster with fewer real steps")
print("  Model-free RL (plain Q-learning) needs more real environment interaction to reach the same policy quality")
print("  When the environment changes (hazard position shifts), a model-based agent must also re-learn its model, adding overhead")
print("  Model-free methods are simpler to implement and more robust when the true dynamics are hard to model accurately")

'''
Output:
Model-Free vs Model-Based RL in a Dynamic Environment
State Space: 6, Goal: 5

   EnvChange  Episodes      Approach  SuccessRate%  AvgStepsToGoal
----------------------------------------------------------------------
    hazard=3         4    model-free           0.0            0.00
    hazard=3         4   model-based         100.0            5.88
    hazard=2         4    model-free           0.0            0.00
    hazard=2         4   model-based         100.0            5.91

Evaluation Summary:
  Model-based RL reuses a learned transition model for extra planning updates, often learning faster with fewer real steps
  Model-free RL (plain Q-learning) needs more real environment interaction to reach the same policy quality
  When the environment changes (hazard position shifts), a model-based agent must also re-learn its model, adding overhead
  Model-free methods are simpler to implement and more robust when the true dynamics are hard to model accurately
'''
