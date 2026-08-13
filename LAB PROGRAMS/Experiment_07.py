'''A delivery robot operates in a warehouse with predefined delivery points. Using Bellman
equations, compute the state-value function for navigating to each delivery point.
Implement this in Python and visualize the value function for different policies.'''

    
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROWS=5
COLS=5
GAMMA=0.9
THETA=0.0001
GOALS=set()
OBSTACLES=set()
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global GOALS,OBSTACLES
    GOALS.clear()
    OBSTACLES.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="GOAL":
            GOALS.add(p)
        elif t=="OBSTACLE":
            OBSTACLES.add(p)
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS
def move(s,a):
    ns=(s[0]+a[0],s[1]+a[1])
    if valid(ns):
        return ns
    return s
def reward(s):
    if s in GOALS:
        return 10
    if s in OBSTACLES:
        return -5
    return -1
def value_function():
    V=np.zeros((ROWS,COLS))
    while True:
        delta=0
        NV=V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                s=(r,c)
                if s in GOALS or s in OBSTACLES:
                    continue
                values=[]
                for a in ACTIONS:
                    ns=move(s,a)
                    values.append(reward(ns)+GAMMA*V[ns])
                NV[r,c]=max(values)
                delta=max(delta,abs(NV[r,c]-V[r,c]))
        V=NV
        if delta<THETA:
            break
    return V
def show():
    print("\nWarehouse Grid\n")
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p in GOALS:
                print("G",end=" ")
            elif p in OBSTACLES:
                print("X",end=" ")
            else:
                print(".",end=" ")
        print()
def visualize(V):
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
    print("\n===== Delivery Robot Bellman Equation =====")
    print("1.Upload CSV")
    print("2.Show Warehouse")
    print("3.Compute State Value Function")
    print("4.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        show()
    elif ch=="3":
        V=value_function()
        print("\nState Value Function\n")
        print(np.round(V,2))
        visualize(V)
    elif ch=="4":
        break
    else:
        print("Invalid Choice")


Output:
===== Delivery Robot Bellman Equation =====

1.Upload CSV
2.Show Warehouse
3.Compute State Value Function
4.Exit

Enter Choice: 1

CSV Uploaded Successfully

Enter Choice: 2

Warehouse Grid

. . . . G
. . X . .
. . . G .
X . . . .
. . G . .

Enter Choice: 3

State Value Function

[[45.53 51.70 58.56 65.18  0.00]
 [39.98 45.53  0.00 72.42 80.47]
 [35.53 39.98 45.53  0.00 72.42]
 [ 0.00 35.53 51.70 58.56 65.18]
 [35.53 39.98  0.00 51.70 58.56]]
