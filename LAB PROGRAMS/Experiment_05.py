from google.colab import files
import pandas as pd
import numpy as np
ROWS=5
COLS=5
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
GAMMA=0.9
THETA=0.0001
PICKUP=(4,4)
TRAFFIC=set()
HOTSPOTS=set()
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global PICKUP,TRAFFIC,HOTSPOTS
    TRAFFIC.clear()
    HOTSPOTS.clear()
    data=pd.read_csv(file)
    for _,r in data.iterrows():
        p=(int(r["Row"]),int(r["Column"]))
        t=str(r["Type"]).upper()
        if t=="PICKUP":
            PICKUP=p
        elif t=="TRAFFIC":
            TRAFFIC.add(p)
        elif t=="HOTSPOT":
            HOTSPOTS.add(p)
def valid(s):
    return 0<=s[0]<ROWS and 0<=s[1]<COLS
def move(s,a):
    ns=(s[0]+a[0],s[1]+a[1])
    if valid(ns):
        return ns
    return s
def reward(s):
    if s==PICKUP:
        return 10
    if s in HOTSPOTS:
        return 3
    if s in TRAFFIC:
        return -5
    return -1
def value_iteration():
    V=np.zeros((ROWS,COLS))
    while True:
        delta=0
        NV=V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                s=(r,c)
                if s==PICKUP or s in TRAFFIC:
                    continue
                best=-99999
                for a in ACTIONS:
                    ns=move(s,a)
                    val=reward(ns)+GAMMA*V[ns]
                    if val>best:
                        best=val
                NV[r,c]=best
                delta=max(delta,abs(NV[r,c]-V[r,c]))
        V=NV
        if delta<THETA:
            break
    policy=np.full((ROWS,COLS)," ",dtype=object)
    d={0:"↑",1:"↓",2:"←",3:"→"}
    for r in range(ROWS):
        for c in range(COLS):
            s=(r,c)
            if s==PICKUP:
                policy[r,c]="P"
            elif s in TRAFFIC:
                policy[r,c]="X"
            else:
                best=-99999
                act=0
                for i,a in enumerate(ACTIONS):
                    ns=move(s,a)
                    val=reward(ns)+GAMMA*V[ns]
                    if val>best:
                        best=val
                        act=i
                policy[r,c]=d[act]
    return V,policy
def show():
    print("\nTaxi Dispatch Grid\n")
    for r in range(ROWS):
        for c in range(COLS):
            p=(r,c)
            if p==PICKUP:
                print("P",end=" ")
            elif p in TRAFFIC:
                print("X",end=" ")
            elif p in HOTSPOTS:
                print("H",end=" ")
            else:
                print(".",end=" ")
        print()
while True:
    print("\n====== Taxi Dispatch Value Iteration ======")
    print("1.Upload CSV")
    print("2.Show Grid")
    print("3.Run Value Iteration")
    print("4.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
    elif ch=="2":
        show()
    elif ch=="3":
        V,P=value_iteration()
        print("\nOptimal Value Function\n")
        print(np.round(V,2))
        print("\nOptimal Policy\n")
        print(P)
        pd.DataFrame(np.round(V,2)).to_csv("value_iteration_output.csv",index=False)
        print("\nOutput CSV Generated")
        files.download("value_iteration_output.csv")
    elif ch=="4":
        break
    else:
        print("Invalid Choice")


Output:
====== Taxi Dispatch Value Iteration ======
1.Upload CSV
2.Show Grid
3.Run Value Iteration
4.Exit
Enter Choice: 2

Taxi Dispatch Grid

. . . . H 
. . X . . 
. . X . H 
. X . . . 
. H . . P 

====== Taxi Dispatch Value Iteration ======
1.Upload CSV
2.Show Grid
3.Run Value Iteration
4.Exit
Enter Choice: 3

Optimal Value Function

[[19.16 22.4  26.   30.   30.  ]
 [19.16 19.16  0.   26.   30.  ]
 [22.4  19.16  0.   30.   30.  ]
 [26.    0.   26.   26.   30.  ]
 [30.   30.   30.   26.    0.  ]]

Optimal Policy

[['→' '→' '→' '→' '↑']
 ['↓' '↑' 'X' '↓' '↓']
 ['↓' '←' 'X' '→' '→']
 ['↓' 'X' '↓' '↑' '↑']
 ['→' '↓' '←' '←' 'P']]

Output CSV Generated
