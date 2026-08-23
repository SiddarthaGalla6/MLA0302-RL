'''Question: An online advertising system uses Reinforcement Learning to maximize click-
through rates. Outline reward design challenges and explain their impact on system
performance.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_ADS = 6
true_ctr = np.array([0.05, 0.12, 0.08, 0.20, 0.03, 0.15])
true_conversion = np.array([0.01, 0.02, 0.015, 0.005, 0.03, 0.025])
trials = 1500
def run_ad_bandit(reward_mode, epsilon=0.15):
    Q = np.zeros(N_ADS)
    counts = np.zeros(N_ADS)
    total_clicks, total_conversions, total_reward = 0, 0, 0.0
    for t in range(trials):
        eps = max(0.08, epsilon * (1 - t / trials))
        if random.random() < eps:
            ad = random.randint(0, N_ADS - 1)
        else:
            ad = int(np.argmax(Q))
        clicked = np.random.binomial(1, true_ctr[ad])
        converted = clicked and np.random.binomial(1, true_conversion[ad])
        if reward_mode == 'click_only':
            reward = 1.0 if clicked else 0.0
        else:
            reward = (1.0 if clicked else 0.0) + (25.0 if converted else 0.0)
        counts[ad] += 1
        Q[ad] += (reward - Q[ad]) / counts[ad]
        total_clicks += clicked
        total_conversions += int(converted)
        total_reward += reward
    return total_clicks, total_conversions, total_reward, Q
print("Online Advertising RL: Click-Through Rate Optimization")
print(f"Number of Ads: {N_ADS}, Trials: {trials}")
print(f"True CTR per ad: {true_ctr}")
print(f"True conversion rate per ad: {true_conversion}")
print(f"\n{'RewardDesign':>15} {'TotalClicks':>12} {'TotalConversions':>17} {'BestAdChosen':>13}")
print("-" * 62)
for mode in ['click_only', 'click_and_conversion']:
    clicks, conversions, total_reward, Q = run_ad_bandit(mode)
    best_ad = int(np.argmax(Q))
    print(f"{mode:>15} {clicks:>12} {conversions:>17} {best_ad:>13}")
print("\nTrue highest-CTR ad:", int(np.argmax(true_ctr)))
print("True highest-conversion ad:", int(np.argmax(true_conversion)))
print("\nReward Design Challenges:")
print("  Optimizing for clicks alone favors ads that attract clicks but may rarely convert to real value")
print("  With rare conversion events, a small trial budget makes it hard to reliably detect the true best ad")
print("  Adding conversion value nudges the reward signal toward business outcomes, not just engagement")
print("  Reward design directly determines what the system treats as success, so misaligned rewards degrade real performance")

'''
Output:
Online Advertising RL: Click-Through Rate Optimization
Number of Ads: 6, Trials: 1500
True CTR per ad: [0.05 0.12 0.08 0.2  0.03 0.15]
True conversion rate per ad: [0.01  0.02  0.015 0.005 0.03  0.025]

   RewardDesign  TotalClicks  TotalConversions  BestAdChosen
--------------------------------------------------------------
     click_only          285                 2             3
click_and_conversion          271                 3             3

True highest-CTR ad: 3
True highest-conversion ad: 4

Reward Design Challenges:
  Optimizing for clicks alone favors ads that attract clicks but may rarely convert to real value
  With rare conversion events, a small trial budget makes it hard to reliably detect the true best ad
  Adding conversion value nudges the reward signal toward business outcomes, not just engagement
  Reward design directly determines what the system treats as success, so misaligned rewards degrade real performance
'''
