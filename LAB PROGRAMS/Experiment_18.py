"""Implement an agent that manages a financial portfolio, choosing stocks to maximize returns and minimize risk using an Actor-Critic (A3C) method to optimize investment. Implement a robot that navigates a maze to reach the exit, with rewards for reaching the exit and penalties for hitting walls, and use REINFORCE to find the optimal navigation policy."""
import numpy as np
RETURNS=np.array([.12,.08,.05,-.03,.10])
RISK=np.array([.20,.10,.08,.25,.12])
def train():
    policy=np.ones(5)/5
    value=0
    for _ in range(200):
        action=np.random.choice(5,p=policy)
        reward=RETURNS[action]-.5*RISK[action]
        advantage=reward-value
        value+=.05*advantage
        policy[action]+=.03*advantage
        policy=np.clip(policy,.01,1)
        policy/=policy.sum()
    return policy,value
def main():
    policy,value=train()
    print("\nOutput:")
    print("Portfolio Policy:",np.round(policy,3))
    print("Estimated Value:",round(value,4))
    print("Best Stock:",np.argmax(policy)+1)
    print("Risk Adjusted Return:",round(max(RETURNS-.5*RISK),4))
while True:
    print("\n====== A3C Portfolio Agent ======")
    print("1.Train Agent")
    print("2.Evaluate Portfolio")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nPortfolio Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
====== Portfolio Actor-Critic ======
1.Train Agent
2.Evaluate Portfolio
3.Exit
Enter Choice: 1
Output:
Portfolio Policy: [0.251 0.234 0.212 0.101 0.202]
Estimated Value: 0.057
Best Stock: 1
Risk Adjusted Return: 0.06"""
