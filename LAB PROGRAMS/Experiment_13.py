from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROWS=5
COLS=5
ALPHA=0.1
GAMMA=0.9
EPSILON=0.2
EPISODES=300
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
DIR=["↑","↓","←","→"]
START=(0,0)
GOAL=(4,4)
FOOD=set()
GHOSTS=set()
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global START,GOAL,FOOD,GHOSTS
    FOOD.clear()
    GHOSTS.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="START":
            START=p
        elif t=="GOAL":
            GOAL=p
        elif t=="FOOD":
            FOOD.add(p)
        elif t=="GHOST":
            GHOSTS.add(p)
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS
def move(s,a):
    ns=(s[0]+ACTIONS[a][0],s[1]+ACTIONS[a][1])
    if valid(ns):
        return ns
    return s
def reward(s):
    if s==GOAL:
        return 20
    if s in FOOD:
        return 10
    if s in GHOSTS:
        return -15
    return -1
def choose(Q,s):
    if np.random.rand()<EPSILON:
        return np.random.randint(4)
    return np.argmax(Q[s[0],s[1]])
def qlearning():
    Q=np.zeros((ROWS,COLS,4))
    history=[]
    for ep in range(EPISODES):
        s=START
        total=0
        while s!=GOAL:
            a=choose(Q,s)
            ns=move(s,a)
            r=reward(ns)
            Q[s[0],s[1],a]+=ALPHA*(r+GAMMA*np.max(Q[ns[0],ns[1]])-Q[s[0],s[1],a])
            s=ns
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
            if p==START:
                P[r,c]="S"
            elif p==GOAL:
                P[r,c]="G"
            elif p in FOOD:
                P[r,c]="F"
            elif p in GHOSTS:
                P[r,c]="X"
            else:
                P[r,c]=DIR[np.argmax(Q[r,c])]
    return P
def graph(h):
    plt.figure(figsize=(7,4))
    plt.plot(h)
    plt.title("Q-Learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()
while True:
    print("\n====== Pac-Man Q-Learning ======")
    print("1.Upload CSV")
    print("2.Train Agent")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        Q,H=qlearning()
        P=policy(Q)
        print("\nLearned Policy\n")
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
====== Pac-Man Q-Learning ======

1.Upload CSV
2.Train Agent
3.Exit

Enter Choice: 1

CSV Uploaded Successfully

Enter Choice: 2

Learned Policy

[['S' '→' '→' '→' 'F']
 ['↓' '↓' '→' 'X' '↓']
 ['↓' '→' 'F' '→' '↓']
 ['↓' 'X' '→' '→' '↓']
 ['→' '→' '→' '→' 'G']]

Maximum Reward : 52
Average Reward : 38.64
Final Reward : 50
