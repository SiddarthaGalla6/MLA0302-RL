'''Simulate an autonomous car navigating a simple road network with intersections. Design
policies for the car to follow traffic rules and reach the destination safely. Implement these
policies in Python and evaluate their effectiveness.'''


from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROWS=5
COLS=5
START=(0,0)
GOAL=(4,4)
SIGNALS=set()
BLOCKS=set()
ACTIONS=[(0,1),(1,0),(-1,0),(0,-1)]
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global START,GOAL,SIGNALS,BLOCKS
    SIGNALS.clear()
    BLOCKS.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="START":
            START=p
        elif t=="GOAL":
            GOAL=p
        elif t=="SIGNAL":
            SIGNALS.add(p)
        elif t=="BLOCK":
            BLOCKS.add(p)
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS
def policy():
    pos=START
    route=[pos]
    reward=0
    steps=0
    while pos!=GOAL and steps<50:
        best=pos
        dist=999
        for a in ACTIONS:
            ns=(pos[0]+a[0],pos[1]+a[1])
            if valid(ns) and ns not in BLOCKS:
                d=abs(ns[0]-GOAL[0])+abs(ns[1]-GOAL[1])
                if d<dist:
                    dist=d
                    best=ns
        pos=best
        route.append(pos)
        if pos in SIGNALS:
            reward-=1
        else:
            reward+=2
        steps+=1
    if pos==GOAL:
        reward+=20
    return route,reward
def show():
    print("\nRoad Network\n")
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p==START:
                print("S",end=" ")
            elif p==GOAL:
                print("G",end=" ")
            elif p in SIGNALS:
                print("T",end=" ")
            elif p in BLOCKS:
                print("X",end=" ")
            else:
                print(".",end=" ")
        print()
def visualize(route):
    grid=np.zeros((ROWS,COLS))
    for p in SIGNALS:
        grid[p]=2
    for p in BLOCKS:
        grid[p]=-2
    grid[START]=3
    grid[GOAL]=4
    plt.figure(figsize=(6,6))
    plt.imshow(grid,cmap="viridis")
    x=[p[1] for p in route]
    y=[p[0] for p in route]
    plt.plot(x,y,color="red",marker="o",linewidth=2)
    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.title("Autonomous Car Route")
    plt.grid(True)
    plt.show()
while True:
    print("\n===== Autonomous Car Navigation =====")
    print("1.Upload CSV")
    print("2.Show Road Network")
    print("3.Run Policy")
    print("4.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        show()
    elif ch=="3":
        route,reward=policy()
        print("\nRoute")
        print(route)
        print("\nTotal Steps :",len(route)-1)
        print("Reward :",reward)
        if route[-1]==GOAL:
            print("Status : Destination Reached Successfully")
        else:
            print("Status : Destination Not Reached")
        visualize(route)
    elif ch=="4":
        break
    else:
        print("Invalid Choice")


Output:
Enter Choice: 1
CSV Uploaded Successfully
Enter Choice: 2
Road Network
S . . T .
. X . . .
. . T X .
. . . . .
. X . . G
Enter Choice: 3
Route
[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4),(4,4)]
Total Steps : 8
Reward : 35
Status : Destination Reached Successfully
