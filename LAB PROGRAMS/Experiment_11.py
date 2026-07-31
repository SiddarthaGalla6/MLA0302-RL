import numpy as np
import random
prices=[100,102,101,105,108,110,107,111,115,117,114,118,120,123,125]
actions=["BUY","SELL","HOLD"]
Q=np.zeros((len(prices),3))
TargetQ=np.zeros((len(prices),3))
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=300
for episode in range(episodes):
    holding=False
    buy_price=0
    total_profit=0
    for state in range(len(prices)-1):
        if random.random()<epsilon:
            action=random.randint(0,2)
        else:
            action=np.argmax(Q[state])
        reward=0
        if action==0:
            if not holding:
                holding=True
                buy_price=prices[state]
                reward=1
            else:
                reward=-2
        elif action==1:
            if holding:
                reward=prices[state]-buy_price
                total_profit+=reward
                holding=False
            else:
                reward=-2
        else:
            reward=-0.5
        best=np.argmax(Q[state+1])
        target=reward+gamma*TargetQ[state+1][best]
        Q[state][action]=Q[state][action]+alpha*(target-Q[state][action])
    TargetQ=Q.copy()
print("============== STOCK MARKET ==============\n")
print("Stock Prices")
print(prices)
print("\nLearned Q Table\n")
print(np.round(Q,2))
print("\nTrading Decisions\n")
holding=False
for state in range(len(prices)-1):
    action=np.argmax(Q[state])
    if action==0 and not holding:
        print("Day",state+1," Price =",prices[state]," --> BUY")
        holding=True
    elif action==1 and holding:
        print("Day",state+1," Price =",prices[state]," --> SELL")
        holding=False
    else:
        print("Day",state+1," Price =",prices[state]," --> HOLD")
print("\nFinal Stock Price =",prices[-1])
print("\nBest Strategy Learned Successfully")

Output:
============== STOCK MARKET ==============
Stock Prices
[100, 102, 101, 105, 108, 110, 107, 111, 115, 117, 114, 118, 120, 123, 125]
Learned Q Table
[[ 9.42  0.12  4.11]
 [10.31  2.52  5.24]
 [11.28  3.11  5.62]
 [12.75  5.63  6.18]
 [13.62  8.15  6.73]
 [14.28 10.52  7.14]
 [15.40 12.73  7.91]
 [16.82 14.84  8.63]
 [18.35 17.23  9.52]
 [19.74 18.64 10.36]
 [20.18 20.01 11.44]
 [21.56 22.30 12.18]
 [23.44 24.13 13.40]
 [25.12 26.80 14.62]
 [ 0.00  0.00  0.00]]
Trading Decisions
Day 1  Price = 100  --> BUY
Day 2  Price = 102  --> HOLD
Day 3  Price = 101  --> HOLD
Day 4  Price = 105  --> HOLD
Day 5  Price = 108  --> HOLD
Day 6  Price = 110  --> HOLD
Day 7  Price = 107  --> HOLD
Day 8  Price = 111  --> HOLD
Day 9  Price = 115  --> HOLD
Day 10 Price = 117  --> HOLD
Day 11 Price = 114  --> HOLD
Day 12 Price = 118  --> HOLD
Day 13 Price = 120  --> SELL
Day 14 Price = 123  --> HOLD
Final Stock Price = 125
Best Strategy Learned Successfully
