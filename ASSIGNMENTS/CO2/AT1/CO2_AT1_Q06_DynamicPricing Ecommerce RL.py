'''Question: Apply Reinforcement Learning to dynamic pricing in e-commerce. Define the state
space, action space, and reward function. Discuss challenges in balancing
exploration and exploitation.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
INVENTORY_BINS = 4
DEMAND_BINS = 4
COMPETITOR_BINS = 3
N_STATES = INVENTORY_BINS * DEMAND_BINS * COMPETITOR_BINS
PRICE_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2]
ACTIONS = [f"price_x{p}" for p in PRICE_MULTIPLIERS]
N_ACTIONS = len(ACTIONS)
BASE_PRICE = 50.0
BASE_COST = 30.0
def encode_state(inventory, demand, competitor):
    return inventory * DEMAND_BINS * COMPETITOR_BINS + demand * COMPETITOR_BINS + competitor
def simulate_sale(price_multiplier, demand, competitor_level):
    price = BASE_PRICE * price_multiplier
    competitor_price = BASE_PRICE * (0.9 + 0.1 * competitor_level)
    price_ratio = price / competitor_price
    base_prob = 0.3 + 0.15 * demand
    sale_prob = max(0.02, min(0.95, base_prob - 0.4 * (price_ratio - 1)))
    sold = random.random() < sale_prob
    return sold, price
def compute_pricing_reward(sold, price, inventory):
    if not sold:
        return -1.0
    profit = price - BASE_COST
    stockout_bonus = 2.0 if inventory <= 1 else 0.0
    overstock_penalty = -1.5 if inventory >= 3 else 0.0
    return profit * 0.2 + stockout_bonus + overstock_penalty
def run_pricing_rl(epsilon, n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma = 0.15, 0.9
    total_profit, total_sales = 0.0, 0
    for ep in range(n_episodes):
        inventory = random.randint(0, INVENTORY_BINS - 1)
        demand = random.randint(0, DEMAND_BINS - 1)
        competitor = random.randint(0, COMPETITOR_BINS - 1)
        for step in range(10):
            state = encode_state(inventory, demand, competitor)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            sold, price = simulate_sale(PRICE_MULTIPLIERS[action], demand, competitor)
            reward = compute_pricing_reward(sold, price, inventory)
            if sold:
                inventory = max(inventory - 1, 0)
                total_profit += price - BASE_COST
                total_sales += 1
            demand = random.randint(0, DEMAND_BINS - 1)
            competitor = random.randint(0, COMPETITOR_BINS - 1)
            next_state = encode_state(inventory, demand, competitor)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    return total_profit, total_sales
print("Dynamic Pricing RL for E-Commerce")
print(f"State Space: inventory_level x demand_level x competitor_price_level = {N_STATES}")
print(f"Actions (price multipliers): {ACTIONS}")
print(f"{'Epsilon':>8} {'TotalProfit':>12} {'TotalSales':>11}")
print("-" * 35)
for eps in [0.05, 0.2, 0.4]:
    profit, sales = run_pricing_rl(eps)
    print(f"{eps:>8} {profit:>12.2f} {sales:>11}")
print("\nMDP Component Summary:")
print("  States : inventory level x customer demand level x competitor price level")
print("  Actions:", ACTIONS)
print("  Reward : scaled profit per sale + stockout bonus - overstock penalty - no-sale penalty")
print("\nExploration vs Exploitation Challenge:")
print("  Low epsilon exploits known profitable prices quickly but may miss better price points")
print("  High epsilon tests more prices, risking short-term revenue loss from suboptimal pricing")
print("  Non-stationary demand and competitor behavior require sustained exploration, not a one-time search")

'''
Output:
Dynamic Pricing RL for E-Commerce
State Space: inventory_level x demand_level x competitor_price_level = 48
Actions (price multipliers): ['price_x0.8', 'price_x0.9', 'price_x1.0', 'price_x1.1', 'price_x1.2']
 Epsilon  TotalProfit  TotalSales
-----------------------------------
    0.05     37185.00        2818
     0.2     43530.00        2821
     0.4     49010.00        2646

MDP Component Summary:
  States : inventory level x customer demand level x competitor price level
  Actions: ['price_x0.8', 'price_x0.9', 'price_x1.0', 'price_x1.1', 'price_x1.2']
  Reward : scaled profit per sale + stockout bonus - overstock penalty - no-sale penalty

Exploration vs Exploitation Challenge:
  Low epsilon exploits known profitable prices quickly but may miss better price points
  High epsilon tests more prices, risking short-term revenue loss from suboptimal pricing
  Non-stationary demand and competitor behavior require sustained exploration, not a one-time search
'''
