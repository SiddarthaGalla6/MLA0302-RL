import numpy as np
ROWS,COLS=5,5
START=(0,0)
GOAL=(4,4)
FOOD={(1,2),(2,4),(3,1)}
GHOST={(2,2),(3,3)}
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
Q=np.zeros((ROWS,COLS,4))
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000
def valid(r,c):
    return 0<=r<ROWS and 0<=c<COLS
def choose(state):
    if np.random.rand()<epsilon:
        return np.random.randint(4)
    return np.argmax(Q[state[0],state[1]])
for ep in range(episodes):
    state=START
    while state!=GOAL:
        action=choose(state)
        dr,dc=ACTIONS[action]
        nr,nc=state[0]+dr,state[1]+dc
        if not valid(nr,nc):
            nr,nc=state
        next_state=(nr,nc)
        reward=-1
        if next_state in FOOD:
            reward=5
        if next_state in GHOST:
            reward=-10
        if next_state==GOAL:
            reward=20
        best=np.max(Q[next_state[0],next_state[1]])
        Q[state[0],state[1],action]+=alpha*(reward+gamma*best-Q[state[0],state[1],action])
        state=next_state
print("Grid Game")
for r in range(ROWS):
    row=[]
    for c in range(COLS):
        cell=(r,c)
        if cell==START:
            row.append("S")
        elif cell==GOAL:
            row.append("G")
        elif cell in FOOD:
            row.append("F")
        elif cell in GHOST:
            row.append("X")
        else:
            row.append(".")
    print(" ".join(row))
print("\nLearned Path")
state=START
path=[state]
score=0
visited=set()
while state!=GOAL:
    visited.add(state)
    action=np.argmax(Q[state[0],state[1]])
    dr,dc=ACTIONS[action]
    nr,nc=state[0]+dr,state[1]+dc
    if not valid(nr,nc) or (nr,nc) in visited:
        break
    state=(nr,nc)
    path.append(state)
    if state in FOOD:
        score+=5
    elif state in GHOST:
        score-=10
    else:
        score-=1
    if state==GOAL:
        score+=20
for p in path:
    print(p)
print("\nSteps :",len(path)-1)
print("Final Score :",score)
print("Goal Reached :",state==GOAL)

Output:
Grid Game

S . . . .
. . F . .
. . X . F
. F . X .
. . . . G

Learned Path
(0, 0)
(0, 1)
(0, 2)
(1, 2)
(1, 3)
(1, 4)
(2, 4)
(3, 4)
(4, 4)

Steps : 8
Final Score : 28
Goal Reached : True
