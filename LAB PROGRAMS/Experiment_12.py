'''A robot vacuum cleaner navigates a house with various rooms and obstacles. Use the SARSA
algorithm to learn the optimal cleaning policy that maximizes the cleaned area while
minimizing energy usage. Implement this in Python.'''

    
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROWS=5
COLS=5
ALPHA=0.1
GAMMA=0.9
EPSILON=0.2
EPISODES=200
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
DIR=["↑","↓","←","→"]
START=(0,0)
GOAL=(4,4)
DIRTY=set()
OBSTACLES=set()
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global START,GOAL,DIRTY,OBSTACLES
    DIRTY.clear()
    OBSTACLES.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="START":
            START=p
        elif t=="GOAL":
            GOAL=p
        elif t=="DIRT":
            DIRTY.add(p)
        elif t=="OBSTACLE":
            OBSTACLES.add(p)
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS and s not in OBSTACLES
def move(s,a):
    ns=(s[0]+ACTIONS[a][0],s[1]+ACTIONS[a][1])
    if valid(ns):
        return ns
    return s
def reward(s):
    if s==GOAL:
        return 20
    if s in DIRTY:
        return 10
    return -1
def choose(Q,s):
    if np.random.rand()<EPSILON:
        return np.random.randint(4)
    return np.argmax(Q[s[0],s[1]])
def sarsa():
    Q=np.zeros((ROWS,COLS,4))
    history=[]
    for ep in range(EPISODES):
        s=START
        a=choose(Q,s)
        total=0
        while s!=GOAL:
            ns=move(s,a)
            r=reward(ns)
            na=choose(Q,ns)
            Q[s[0],s[1],a]+=ALPHA*(r+GAMMA*Q[ns[0],ns[1],na]-Q[s[0],s[1],a])
            s=ns
            a=na
            total+=r
            if total<-300:
                break
        history.append(total)
    return Q,history
def policy(Q):
    P=np.full((ROWS,COLS)," ",dtype=object)
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p==GOAL:
                P[r,c]="G"
            elif p==START:
                P[r,c]="S"
            elif p in DIRTY:
                P[r,c]="D"
            elif p in OBSTACLES:
                P[r,c]="X"
            else:
                P[r,c]=DIR[np.argmax(Q[r,c])]
    return P
def graph(h):
    plt.figure(figsize=(7,4))
    plt.plot(h)
    plt.title("SARSA Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()
while True:
    print("\n====== Robot Vacuum SARSA ======")
    print("1.Upload CSV")
    print("2.Train SARSA Agent")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        Q,H=sarsa()
        P=policy(Q)
        print("\nLearned Cleaning Policy\n")
        print(P)
        print("\nMaximum Reward :",max(H))
        print("Average Reward :",round(np.mean(H),2))
        print("Final Reward :",H[-1])
        graph(H)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")


Output:
====== Robot Vacuum SARSA ======

1.Upload CSV
2.Train SARSA Agent
3.Exit

Enter Choice: 1

CSV Uploaded Successfully

Enter Choice: 2

Learned Cleaning Policy

[['S' '→' '→' '→' 'D']
 ['↓' '↓' 'X' '→' '↓']
 ['↓' 'D' '→' 'X' '↓']
 ['↓' '→' '→' '→' 'D']
 ['→' '→' '→' '→' 'G']]

Maximum Reward : 48
Average Reward : 35.72
Final Reward : 46
