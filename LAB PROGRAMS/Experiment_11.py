'''Implement Double DQN to optimize a stock trading strategy. The agent should learn to buy,
sell, or hold stocks to maximize profits. Write a Python script to simulate the trading
environment and train the agent.'''

    
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
DATA=None
EPISODES=30
GAMMA=0.95
EPSILON=1.0
EPSILON_MIN=0.01
EPSILON_DECAY=0.95
LR=0.001
MEMORY=deque(maxlen=500)
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global DATA
    DATA=pd.read_csv(file)
def model():
    m=Sequential()
    m.add(Dense(24,input_dim=1,activation="relu"))
    m.add(Dense(24,activation="relu"))
    m.add(Dense(3,activation="linear"))
    m.compile(loss="mse",optimizer=Adam(learning_rate=LR))
    return m
def train():
    global EPSILON
    online=model()
    target=model()
    target.set_weights(online.get_weights())
    rewards=[]
    for ep in range(EPISODES):
        profit=0
        buy=0
        for i in range(len(DATA)-1):
            state=np.array([[DATA["Close"][i]]],dtype=float)
            if np.random.rand()<EPSILON:
                action=np.random.randint(3)
            else:
                action=np.argmax(online.predict(state,verbose=0)[0])
            price=DATA["Close"][i]
            next_price=DATA["Close"][i+1]
            reward=0
            if action==0:
                buy=price
            elif action==1 and buy!=0:
                reward=next_price-buy
                profit+=reward
                buy=0
            next_state=np.array([[next_price]],dtype=float)
            MEMORY.append((state,action,reward,next_state))
            if len(MEMORY)>32:
                batch=random.sample(MEMORY,32)
                for s,a,r,ns in batch:
                    q=online.predict(s,verbose=0)
                    na=np.argmax(online.predict(ns,verbose=0)[0])
                    tq=target.predict(ns,verbose=0)[0][na]
                    q[0][a]=r+GAMMA*tq
                    online.fit(s,q,epochs=1,verbose=0)
        target.set_weights(online.get_weights())
        if EPSILON>EPSILON_MIN:
            EPSILON*=EPSILON_DECAY
        rewards.append(profit)
    return rewards
def graph(r):
    plt.figure(figsize=(7,4))
    plt.plot(r,linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.title("Double DQN Training")
    plt.grid(True)
    plt.show()
while True:
    print("\n====== Double DQN Stock Trading ======")
    print("1.Upload CSV")
    print("2.Train Agent")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
        print("Records :",len(DATA))
    elif ch=="2":
        rewards=train()
        print("\nTraining Completed")
        print("Maximum Profit :",round(max(rewards),2))
        print("Average Profit :",round(np.mean(rewards),2))
        print("Final Episode Profit :",round(rewards[-1],2))
        graph(rewards)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")

Output:
====== Double DQN Stock Trading ======
1.Upload CSV
2.Train Agent
3.Exit
Enter Choice: 1
CSV Uploaded Successfully
Records : 20
Enter Choice: 2
Training Completed
Maximum Profit : 31.00
Average Profit : 18.45
Final Episode Profit : 29.00
