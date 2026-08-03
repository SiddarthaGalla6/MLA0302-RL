from google.colab import files
import pandas as pd
import numpy as np
ROWS=5
COLS=5
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
GAMMA=0.9
ITEMS=set()
OBSTACLES=set()
GOAL=(4,4)
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global ITEMS,OBSTACLES,GOAL
    ITEMS.clear()
    OBSTACLES.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="DELIVERY":
            ITEMS.add(p)
        elif t=="OBSTACLE":
            OBSTACLES.add(p)
        elif t=="GOAL":
            GOAL=p
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS
def move(s,a):
    ns=(s[0]+a[0],s[1]+a[1])
    if valid(ns):
        return ns
    return s
def reward(s):
    if s==GOAL:
        return 10
    if s in ITEMS:
        return 5
    if s in OBSTACLES:
        return -5
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
                    ns=move(s,ACTIONS[policy[r,c]])
                    NV[r,c]=reward(ns)+GAMMA*V[ns]
                    delta=max(delta,abs(NV[r,c]-V[r,c]))
            V=NV
            if delta<0.0001:
                break
        stable=True
        for r in range(ROWS):
            for c in range(COLS):
                s=(r,c)
                if s==GOAL or s in OBSTACLES:
                    continue
                old=policy[r,c]
                best=old
                bestv=-999999
                for i,a in enumerate(ACTIONS):
                    ns=move(s,a)
                    val=reward(ns)+GAMMA*V[ns]
                    if val>bestv:
                        bestv=val
                        best=i
                policy[r,c]=best
                if old!=best:
                    stable=False
        if stable:
            break
    return V,policy
def show():
    print("\nCity Grid\n")
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p==GOAL:
                print("G",end=" ")
            elif p in ITEMS:
                print("D",end=" ")
            elif p in OBSTACLES:
                print("X",end=" ")
            else:
                print(".",end=" ")
        print()
while True:
    print("\n===== Delivery Drone Policy Iteration =====")
    print("1.Upload CSV")
    print("2.Show Grid")
    print("3.Run Policy Iteration")
    print("4.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        show()
    elif ch=="3":
        V,P=policy_iteration()
        print("\nOptimal Value Function\n")
        print(np.round(V,2))
        out=pd.DataFrame(np.round(V,2))
        out.to_csv("policy_iteration_output.csv",index=False)
        print("\nOutput CSV Generated")
        files.download("policy_iteration_output.csv")
    elif ch=="4":
        break
    else:
        print("Invalid Choice")


Output :
===== Delivery Drone Policy Iteration =====
1.Upload CSV
2.Show Grid
3.Run Policy Iteration
4.Exit
Enter Choice: 2

City Grid

. . . . D 
. . X . . 
. X . . D 
. . . X . 
. . D . G 

===== Delivery Drone Policy Iteration =====
1.Upload CSV
2.Show Grid
3.Run Policy Iteration
4.Exit
Enter Choice: 3

Optimal Value Function

[[33.74 38.6  44.   50.   50.  ]
 [29.37 33.74  0.   44.   50.  ]
 [33.74  0.   44.   50.   50.  ]
 [38.6  44.   50.    0.   50.  ]
 [44.   50.   50.   50.    0.  ]]

Output CSV Generated
