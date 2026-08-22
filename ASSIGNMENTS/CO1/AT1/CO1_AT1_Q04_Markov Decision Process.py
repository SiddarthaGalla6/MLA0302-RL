'''Question: To develop a simple Markov Decision Process model and simulate state transitions, reward functions, and policy execution for 
understanding sequential decision-making problems.'''


# Code:
import numpy as np
states = ["S0", "S1", "S2", "S3_Terminal"]
actions = ["A0", "A1"]
gamma = 0.9
transition = {
    "S0": {"A0": [("S1", 0.7, 5), ("S0", 0.3, -1)],
           "A1": [("S2", 0.6, 3), ("S0", 0.4, -1)]},
    "S1": {"A0": [("S3_Terminal", 0.8, 10), ("S1", 0.2, 2)],
           "A1": [("S2", 0.5, 4), ("S1", 0.5, 1)]},
    "S2": {"A0": [("S1", 0.6, 3), ("S2", 0.4, 0)],
           "A1": [("S3_Terminal", 0.7, 8), ("S2", 0.3, -2)]},
    "S3_Terminal": {}
}
policy = {"S0": "A0", "S1": "A0", "S2": "A1"}
print("MDP Simulation")
print(f"States: {states}")
print(f"Actions: {actions}")
print(f"Discount Factor (gamma): {gamma}\n")
state = "S0"
total_reward = 0
step = 0
print("Policy Execution Trace:")
while state != "S3_Terminal" and step < 10:
    action = policy.get(state)
    if not action:
        break
    transitions = transition[state][action]
    probs = [t[1] for t in transitions]
    chosen = transitions[np.random.choice(len(transitions), p=probs)]
    next_state, prob, reward = chosen
    discounted = (gamma ** step) * reward
    total_reward += discounted
    print(f"Step {step+1}: {state} --[{action}]--> {next_state} | Reward={reward}, Discounted={discounted:.3f}")
    state = next_state
    step += 1
print(f"\nFinal State: {state}")
print(f"Total Discounted Reward: {total_reward:.3f}")


'''Output:
# MDP Simulation
# States: ['S0', 'S1', 'S2', 'S3_Terminal']
# Actions: ['A0', 'A1']
# Discount Factor (gamma): 0.9
# Policy Execution Trace:
# Step 1: S0 --[A0]--> S1 | Reward=5, Discounted=5.000
# Step 2: S1 --[A0]--> S3_Terminal | Reward=10, Discounted=9.000
# Final State: S3_Terminal
# Total Discounted Reward: 14.000'''
