from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROWS=5
COLS=5
GAMMA=0.9
THETA=0.0001
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
DIR=["↑","↓","←","→"]
START=(0,0)
GOAL=(4,4)
OBSTACLES=set()
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global START,GOAL,OBSTACLES
    OBSTACLES.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="START":
            START=p
        elif t=="GOAL":
            GOAL=p
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
    return -1
def policy_iteration():
    policy=np.zeros((ROWS,COLS),dtype=int)
    V=np.zeros((ROWS,COLS))
    while True:
        while True:
            delta=0
            NV=V.copy()
            for r in range(ROWS):
                for c in range(COLS):
                    s=(r,c)
                    if s==GOAL or s in OBSTACLES:
                        continue
                    ns=move(s,policy[r,c])
                    NV[r,c]=reward(ns)+GAMMA*V[ns]
                    delta=max(delta,abs(NV[r,c]-V[r,c]))
            V=NV
            if delta<THETA:
                break
        stable=True
        for r in range(ROWS):
            for c in range(COLS):
                s=(r,c)
                if s==GOAL or s in OBSTACLES:
                    continue
                old=policy[r,c]
                best=old
                bestv=-9999
                for a in range(4):
                    ns=move(s,a)
                    val=reward(ns)+GAMMA*V[ns]
                    if val>bestv:
                        bestv=val
                        best=a
                policy[r,c]=best
                if old!=best:
                    stable=False
        if stable:
            break
    return V,policy
def show_policy(P):
    G=np.full((ROWS,COLS)," ",dtype=object)
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p==START:
                G[r,c]="S"
            elif p==GOAL:
                G[r,c]="G"
            elif p in OBSTACLES:
                G[r,c]="X"
            else:
                G[r,c]=DIR[P[r,c]]
    print("\nOptimal Policy\n")
    print(G)
def graph(V):
    plt.figure(figsize=(6,5))
    plt.imshow(V,cmap="viridis")
    plt.colorbar(label="State Value")
    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.title("State Value Function")
    for i in range(ROWS):
        for j in range(COLS):
            plt.text(j,i,round(V[i,j],1),ha="center",va="center",color="white")
    plt.show()
while True:
    print("\n====== GridWorld Policy Iteration ======")
    print("1.Upload CSV")
    print("2.Run Policy Iteration")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        V,P=policy_iteration()
        print("\nState Value Function\n")
        print(np.round(V,2))
        show_policy(P)
        graph(V)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")


Output:
====== GridWorld Policy Iteration ======

1.Upload CSV
2.Run Policy Iteration
3.Exit

Enter Choice: 1

CSV Uploaded Successfully

Enter Choice: 2

State Value Function

[[ 5.48  7.20  9.11 11.23 13.59]
 [ 7.20  9.11  0.00 13.59 16.21]
 [ 9.11  0.00 13.59 16.21  0.00]
 [11.23 13.59 16.21  0.00 20.00]
 [13.59 16.21 18.90 20.00  0.00]]

Optimal Policy

[['S' '→' '→' '→' '↓']
 ['↓' '↓' 'X' '→' '↓']
 ['↓' 'X' '↓' '↓' 'X']
 ['→' '→' '↓' 'X' '↓']
 ['→' '→' '→' '→' 'G']]
