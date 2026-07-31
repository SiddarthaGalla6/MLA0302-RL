import numpy as np
ROWS,COLS=5,5
START=(0,0)
GOAL=(4,4)
OBST={(1,2),(2,2),(3,1)}
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
Q=np.zeros((ROWS,COLS,4))
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=500
def valid(r,c):
    return 0<=r<ROWS and 0<=c<COLS and (r,c) not in OBST
def choose(state):
    if np.random.rand()<epsilon:
        return np.random.randint(4)
    return np.argmax(Q[state[0],state[1]])
for ep in range(episodes):
    state=START
    action=choose(state)
    while state!=GOAL:
        dr,dc=ACTIONS[action]
        nr,nc=state[0]+dr,state[1]+dc
        if not valid(nr,nc):
            nr,nc=state
        reward=10 if (nr,nc)==GOAL else -1
        next_state=(nr,nc)
        next_action=choose(next_state)
        Q[state[0],state[1],action]+=alpha*(reward+gamma*Q[next_state[0],next_state[1],next_action]-Q[state[0],state[1],action])
        state=next_state
        action=next_action
print("Vacuum Cleaning Policy\n")
path=[START]
state=START
while state!=GOAL:
    a=np.argmax(Q[state[0],state[1]])
    dr,dc=ACTIONS[a]
    nr,nc=state[0]+dr,state[1]+dc
    if not valid(nr,nc):
        break
    state=(nr,nc)
    path.append(state)
print("Start :",START)
print("Goal :",GOAL)
print("Optimal Path")
for p in path:
    print(p)
print("\nTotal Rooms Cleaned :",len(path))
print("Energy Used :",len(path)-1)

Output:
Vacuum Cleaning Policy
Start : (0, 0)
Goal : (4, 4)
Optimal Path
(0,0)
(0,1)
(0,2)
(0,3)
(0,4)
(1,4)
(2,4)
(3,4)
(4,4)
Total Rooms Cleaned : 9
Energy Used : 8
