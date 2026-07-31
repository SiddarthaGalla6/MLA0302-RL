import numpy as np
ROWS,COLS=5,5
GOAL=(4,4)
OBST={(1,2),(2,2),(3,1)}
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
GAMMA=0.9
THETA=0.001
V=np.zeros((ROWS,COLS))
policy=np.random.randint(4,size=(ROWS,COLS))
def valid(r,c):
    return 0<=r<ROWS and 0<=c<COLS and (r,c) not in OBST
while True:
    while True:
        delta=0
        for r in range(ROWS):
            for c in range(COLS):
                if (r,c)==GOAL or (r,c) in OBST:
                    continue
                dr,dc=ACTIONS[policy[r][c]]
                nr,nc=r+dr,c+dc
                if not valid(nr,nc):
                    nr,nc=r,c
                reward=10 if (nr,nc)==GOAL else -1
                value=reward+GAMMA*V[nr][nc]
                delta=max(delta,abs(value-V[r][c]))
                V[r][c]=value
        if delta<THETA:
            break
    stable=True
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c)==GOAL or (r,c) in OBST:
                continue
            old=policy[r][c]
            best=-999
            act=0
            for i,(dr,dc) in enumerate(ACTIONS):
                nr,nc=r+dr,c+dc
                if not valid(nr,nc):
                    nr,nc=r,c
                reward=10 if (nr,nc)==GOAL else -1
                value=reward+GAMMA*V[nr][nc]
                if value>best:
                    best=value
                    act=i
            policy[r][c]=act
            if old!=act:
                stable=False
    if stable:
        break
print("Grid World")
for r in range(ROWS):
    row=[]
    for c in range(COLS):
        if (r,c)==(0,0):
            row.append("S")
        elif (r,c)==GOAL:
            row.append("G")
        elif (r,c) in OBST:
            row.append("X")
        else:
            row.append(".")
    print(" ".join(row))
print("\nState Value Function")
print(np.round(V,2))
print("\nOptimal Path")
state=(0,0)
path=[state]
visit=set()
while state!=GOAL:
    visit.add(state)
    a=policy[state[0]][state[1]]
    dr,dc=ACTIONS[a]
    nr,nc=state[0]+dr,state[1]+dc
    if not valid(nr,nc) or (nr,nc) in visit:
        break
    state=(nr,nc)
    path.append(state)
for p in path:
    print(p)
print("\nSteps :",len(path)-1)
print("Goal Reached :",state==GOAL)

Output:
Grid World

S . . . .
. . X . .
. . X . .
. X . . .
. . . . G

State Value Function

[[ 1.81  3.12  4.58  6.2   8.  ]
 [ 3.12  4.58  0.    8.   10.  ]
 [ 4.58  6.2   0.   10.   12.  ]
 [ 6.2   0.   10.   12.   14.  ]
 [ 8.   10.   12.   14.    0.  ]]

Optimal Path
(0, 0)
(0, 1)
(0, 2)
(0, 3)
(0, 4)
(1, 4)
(2, 4)
(3, 4)
(4, 4)

Steps : 8
Goal Reached : True
