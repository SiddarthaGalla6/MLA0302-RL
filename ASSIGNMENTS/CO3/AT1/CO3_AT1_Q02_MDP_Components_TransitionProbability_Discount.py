'''Question: Estimate how decision-making problems can be modeled as a Markov Decision
Process (MDP), detailing the components of action space, transition probability,
reward function, and discount factor.'''

# Code:
import numpy as np
np.random.seed(42)
STATES = ['low_stock', 'medium_stock', 'high_stock']
N_STATES = len(STATES)
ACTIONS = ['order_more', 'order_less', 'no_order']
N_ACTIONS = len(ACTIONS)
transition_probs = {
    ('low_stock', 'order_more'): {'low_stock': 0.1, 'medium_stock': 0.5, 'high_stock': 0.4},
    ('low_stock', 'order_less'): {'low_stock': 0.6, 'medium_stock': 0.3, 'high_stock': 0.1},
    ('low_stock', 'no_order'): {'low_stock': 0.8, 'medium_stock': 0.2, 'high_stock': 0.0},
    ('medium_stock', 'order_more'): {'low_stock': 0.0, 'medium_stock': 0.4, 'high_stock': 0.6},
    ('medium_stock', 'order_less'): {'low_stock': 0.3, 'medium_stock': 0.5, 'high_stock': 0.2},
    ('medium_stock', 'no_order'): {'low_stock': 0.2, 'medium_stock': 0.6, 'high_stock': 0.2},
    ('high_stock', 'order_more'): {'low_stock': 0.0, 'medium_stock': 0.2, 'high_stock': 0.8},
    ('high_stock', 'order_less'): {'low_stock': 0.1, 'medium_stock': 0.4, 'high_stock': 0.5},
    ('high_stock', 'no_order'): {'low_stock': 0.2, 'medium_stock': 0.3, 'high_stock': 0.5}
}
reward_function = {
    'low_stock': -5.0,
    'medium_stock': 3.0,
    'high_stock': -2.0
}
def sample_next_state(state, action):
    probs = transition_probs[(state, action)]
    return np.random.choice(list(probs.keys()), p=list(probs.values()))
def value_iteration(gamma, theta=1e-4, max_iter=200):
    V = {s: 0.0 for s in STATES}
    for iteration in range(max_iter):
        delta = 0.0
        new_V = {}
        for s in STATES:
            action_values = []
            for a in ACTIONS:
                probs = transition_probs[(s, a)]
                expected_value = sum(p * (reward_function[s_next] + gamma * V[s_next]) for s_next, p in probs.items())
                action_values.append(expected_value)
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    policy = {}
    for s in STATES:
        action_values = []
        for a in ACTIONS:
            probs = transition_probs[(s, a)]
            expected_value = sum(p * (reward_function[s_next] + gamma * V[s_next]) for s_next, p in probs.items())
            action_values.append(expected_value)
        policy[s] = ACTIONS[int(np.argmax(action_values))]
    return V, policy, iteration
print("Inventory Management Modeled as a Markov Decision Process")
print(f"State Space: {STATES}")
print(f"Action Space: {ACTIONS}")
print(f"Reward Function: {reward_function}")
print(f"\nSample transition from 'low_stock' with action 'order_more': {transition_probs[('low_stock', 'order_more')]}")
sampled = sample_next_state('low_stock', 'order_more')
print(f"Sampled next state from that transition: {sampled}")
print(f"\n{'Gamma':>8} {'V(low_stock)':>13} {'V(medium_stock)':>16} {'V(high_stock)':>14} {'Iterations':>11}")
print("-" * 68)
for gamma in [0.1, 0.5, 0.9]:
    V, policy, iterations = value_iteration(gamma)
    print(f"{gamma:>8} {V['low_stock']:>13.3f} {V['medium_stock']:>16.3f} {V['high_stock']:>14.3f} {iterations:>11}")
    print(f"          Optimal policy: {policy}")
print("\nMDP Component Summary:")
print("  Action Space         : order_more, order_less, no_order")
print("  Transition Probability: P(s'|s,a) defined per state-action pair, e.g. table above")
print("  Reward Function       : fixed reward per resulting stock state, penalizing stockouts and overstock")
print("  Discount Factor (gamma): controls how strongly future rewards influence today's ordering decision")

'''
Output:
Inventory Management Modeled as a Markov Decision Process
State Space: ['low_stock', 'medium_stock', 'high_stock']
Action Space: ['order_more', 'order_less', 'no_order']
Reward Function: {'low_stock': -5.0, 'medium_stock': 3.0, 'high_stock': -2.0}

Sample transition from 'low_stock' with action 'order_more': {'low_stock': 0.1, 'medium_stock': 0.5, 'high_stock': 0.4}
Sampled next state from that transition: medium_stock

   Gamma  V(low_stock)  V(medium_stock)  V(high_stock)  Iterations
--------------------------------------------------------------------
     0.1         0.211            0.424         -0.296           4
          Optimal policy: {'low_stock': 'order_more', 'medium_stock': 'no_order', 'high_stock': 'order_less'}
     0.5         0.317            0.585         -0.223          11
          Optimal policy: {'low_stock': 'order_more', 'medium_stock': 'no_order', 'high_stock': 'order_less'}
     0.9         1.409            1.744          0.827          69
          Optimal policy: {'low_stock': 'order_more', 'medium_stock': 'no_order', 'high_stock': 'order_less'}

MDP Component Summary:
  Action Space         : order_more, order_less, no_order
  Transition Probability: P(s'|s,a) defined per state-action pair, e.g. table above
  Reward Function       : fixed reward per resulting stock state, penalizing stockouts and overstock
  Discount Factor (gamma): controls how strongly future rewards influence today's ordering decision
'''
