'''Question: Analyze the application of Reinforcement Learning in portfolio management in
finance. Discuss advantages over traditional models and challenges such as risk and
delayed rewards.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
TREND_BINS = 3
VOLATILITY_BINS = 3
HOLDING_BINS = 3
N_STATES = TREND_BINS * VOLATILITY_BINS * HOLDING_BINS
ACTIONS = ['buy', 'sell', 'hold']
N_ACTIONS = len(ACTIONS)
def encode_state(trend, volatility, holding):
    return trend * VOLATILITY_BINS * HOLDING_BINS + volatility * HOLDING_BINS + holding
def simulate_market_step(price, trend):
    drift = [-0.01, 0.0, 0.015][trend]
    shock = np.random.normal(0, 0.02)
    new_price = price * (1 + drift + shock)
    new_trend = random.choices(range(TREND_BINS), weights=[0.3, 0.4, 0.3])[0]
    new_volatility = random.choices(range(VOLATILITY_BINS), weights=[0.4, 0.4, 0.2])[0]
    return new_price, new_trend, new_volatility
def compute_portfolio_reward(action, price, prev_price, holding, volatility):
    action_name = ACTIONS[action]
    price_change = (price - prev_price) / prev_price
    if action_name == 'buy':
        risk_penalty = -volatility * 0.5
        return price_change * 10.0 + risk_penalty
    elif action_name == 'sell':
        realized_gain = price_change * 10.0 if holding > 0 else -1.0
        return realized_gain
    else:
        opportunity_cost = -0.2 if holding == 0 and price_change > 0 else 0.0
        return opportunity_cost
def run_portfolio_rl(n_episodes=500):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.1, 0.95, 0.3
    ep_returns = []
    for ep in range(n_episodes):
        price = 100.0
        trend = random.randint(0, TREND_BINS - 1)
        volatility = random.randint(0, VOLATILITY_BINS - 1)
        holding = 0
        portfolio_value = 1000.0
        cash = 1000.0
        shares = 0.0
        for step in range(30):
            state = encode_state(trend, volatility, holding)
            action = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(np.argmax(Q[state]))
            new_price, new_trend, new_volatility = simulate_market_step(price, trend)
            reward = compute_portfolio_reward(action, new_price, price, holding, volatility)
            if ACTIONS[action] == 'buy' and cash >= new_price:
                shares += cash / new_price
                cash = 0.0
                holding = 1
            elif ACTIONS[action] == 'sell' and shares > 0:
                cash += shares * new_price
                shares = 0.0
                holding = 0
            next_state = encode_state(new_trend, new_volatility, holding)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            price, trend, volatility = new_price, new_trend, new_volatility
        portfolio_value = cash + shares * price
        ep_returns.append(portfolio_value - 1000.0)
        epsilon = max(0.05, epsilon * 0.99)
    return ep_returns
print("Portfolio Management RL in Finance")
print(f"State Space: price_trend x volatility x holding_status = {N_STATES}")
print(f"Actions: {ACTIONS}")
returns = run_portfolio_rl()
print(f"\nAvg portfolio return first 50 episodes: {np.mean(returns[:50]):.2f}")
print(f"Avg portfolio return last 50 episodes:  {np.mean(returns[-50:]):.2f}")
print(f"Best episode return: {max(returns):.2f}")
print(f"Worst episode return: {min(returns):.2f}")
print("\nMDP Component Summary:")
print("  States : market trend x volatility level x current holding status")
print("  Actions:", ACTIONS)
print("  Reward : price-change-based gain/loss adjusted by volatility risk penalty")
print("\nAdvantages over Traditional Models and Key Challenges:")
print("  RL learns adaptive buy/sell/hold rules directly from simulated price interaction, unlike static mean-variance models")
print("  Delayed reward: true portfolio gain is only realized on sell, so credit assignment across holding periods is harder")
print("  Risk: exploratory trades during training can incur real losses, requiring careful simulation before live deployment")

'''
Output:
Portfolio Management RL in Finance
State Space: price_trend x volatility x holding_status = 27
Actions: ['buy', 'sell', 'hold']

Avg portfolio return first 50 episodes: 48.46
Avg portfolio return last 50 episodes:  49.61
Best episode return: 460.95
Worst episode return: -256.57

MDP Component Summary:
  States : market trend x volatility level x current holding status
  Actions: ['buy', 'sell', 'hold']
  Reward : price-change-based gain/loss adjusted by volatility risk penalty

Advantages over Traditional Models and Key Challenges:
  RL learns adaptive buy/sell/hold rules directly from simulated price interaction, unlike static mean-variance models
  Delayed reward: true portfolio gain is only realized on sell, so credit assignment across holding periods is harder
  Risk: exploratory trades during training can incur real losses, requiring careful simulation before live deployment
'''
