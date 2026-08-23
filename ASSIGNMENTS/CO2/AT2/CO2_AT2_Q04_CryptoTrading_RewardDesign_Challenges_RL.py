'''Question: A cryptocurrency trading system uses Reinforcement Learning to decide buy, sell,
or hold actions. Outline the challenges in designing reward functions and explain how
poor reward design can affect performance.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
TREND_BINS = 3
VOLATILITY_BINS = 3
N_STATES = TREND_BINS * VOLATILITY_BINS * 2
ACTIONS = ['buy', 'sell', 'hold']
N_ACTIONS = len(ACTIONS)
def encode_state(trend, volatility, holding):
    return trend * VOLATILITY_BINS * 2 + volatility * 2 + holding
def simulate_price_step(price):
    trend = random.choices(range(TREND_BINS), weights=[0.35, 0.3, 0.35])[0]
    drift = [-0.02, 0.0, 0.02][trend]
    shock = np.random.normal(0, 0.05)
    new_price = max(price * (1 + drift + shock), 1.0)
    volatility = random.choices(range(VOLATILITY_BINS), weights=[0.3, 0.4, 0.3])[0]
    return new_price, trend, volatility
def reward_naive(action, price, prev_price, holding):
    if ACTIONS[action] == 'buy':
        return 1.0
    elif ACTIONS[action] == 'sell':
        return 1.0
    else:
        return 0.0
def reward_shaped(action, price, prev_price, holding, volatility):
    price_change = (price - prev_price) / prev_price
    if ACTIONS[action] == 'buy':
        return price_change * 10.0 - volatility * 0.5
    elif ACTIONS[action] == 'sell':
        return price_change * 10.0 if holding else -1.0
    else:
        return -0.1 * abs(price_change) * 10.0
def run_trading_rl(reward_mode, n_episodes=400):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    ep_profits = []
    for ep in range(n_episodes):
        price = 100.0
        trend, volatility = random.randint(0, 2), random.randint(0, 2)
        holding, cash, coins = 0, 1000.0, 0.0
        for step in range(25):
            state = encode_state(trend, volatility, holding)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            new_price, new_trend, new_volatility = simulate_price_step(price)
            if reward_mode == 'naive':
                reward = reward_naive(action, new_price, price, holding)
            else:
                reward = reward_shaped(action, new_price, price, holding, volatility)
            if ACTIONS[action] == 'buy' and cash > 0:
                coins += cash / new_price
                cash = 0.0
                holding = 1
            elif ACTIONS[action] == 'sell' and coins > 0:
                cash += coins * new_price
                coins = 0.0
                holding = 0
            next_state = encode_state(new_trend, new_volatility, holding)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            price, trend, volatility = new_price, new_trend, new_volatility
        portfolio_value = cash + coins * price
        ep_profits.append(portfolio_value - 1000.0)
        epsilon = max(0.05, epsilon * 0.99)
    return ep_profits
print("Cryptocurrency Trading RL: Reward Design Comparison")
print(f"State Space: trend x volatility x holding_status = {N_STATES}")
print(f"Actions: {ACTIONS}")
naive_profits = run_trading_rl('naive')
shaped_profits = run_trading_rl('shaped')
print(f"\n{'Reward Design':>15} {'AvgProfitFirst50':>18} {'AvgProfitLast50':>17}")
print("-" * 55)
print(f"{'Naive (+1 trade)':>15} {np.mean(naive_profits[:50]):>18.2f} {np.mean(naive_profits[-50:]):>17.2f}")
print(f"{'Shaped (profit)':>15} {np.mean(shaped_profits[:50]):>18.2f} {np.mean(shaped_profits[-50:]):>17.2f}")
print("\nReward Design Challenges:")
print("  Naive reward (+1 for any trade) encourages excessive buying/selling regardless of profitability")
print("  Sparse or misaligned rewards teach the agent to game the metric instead of maximizing real returns")
print("  Shaped reward tied to actual price change and volatility risk aligns the policy with true profit")
print("  Poor reward design leads to overtrading, ignoring risk, and unstable or negative portfolio outcomes")

'''
Output:
Cryptocurrency Trading RL: Reward Design Comparison
State Space: trend x volatility x holding_status = 18
Actions: ['buy', 'sell', 'hold']

  Reward Design   AvgProfitFirst50   AvgProfitLast50
-------------------------------------------------------
Naive (+1 trade)             -26.33             14.03
Shaped (profit)             -31.30             17.57

Reward Design Challenges:
  Naive reward (+1 for any trade) encourages excessive buying/selling regardless of profitability
  Sparse or misaligned rewards teach the agent to game the metric instead of maximizing real returns
  Shaped reward tied to actual price change and volatility risk aligns the policy with true profit
  Poor reward design leads to overtrading, ignoring risk, and unstable or negative portfolio outcomes
'''
