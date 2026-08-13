'''An autonomous cleaning robot navigates a 5x5 grid where certain cells contain dirt (reward: +1) and obstacles (penalty: -1). The robot starts at the top-left 
corner and must find an optimal policy to clean the entire grid efficiently. Implement the grid environment as an MDP and write a Python program to simulate the 
robot’s navigation using different policies.'''

import numpy as np
import csv
from google.colab import files

ACTIONS=["Up","Down","Left","Right"]
MOVE={"Up":(-1,0),"Down":(1,0),"Left":(0,-1),"Right":(0,1)}
GAMMA=0.9
STEP_COST=-0.04
DIRT_REWARD=1
OBSTACLE_PENALTY=-1

uploaded=files.upload()
filename=list(uploaded.keys())[0]

with open(filename,"r") as f:
    grid=[row for row in csv.reader(f)]

GRID_SIZE=len(grid)

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if grid[i][j]=="S":
            start=(i,j)

def valid(pos):
    r,c=pos
    return 0<=r<GRID_SIZE and 0<=c<GRID_SIZE
def move(pos,action):
    dr,dc=MOVE[action]
    nr,nc=pos[0]+dr,pos[1]+dc
    if not valid((nr,nc)):
        return pos
    if grid[nr][nc]=="#":
        return pos
    return (nr,nc)
def reward(pos):
    r,c=pos
    if grid[r][c]=="D":
        return DIRT_REWARD
    if grid[r][c]=="#":
        return OBSTACLE_PENALTY
    return STEP_COST

V=np.zeros((GRID_SIZE,GRID_SIZE))
policy=[["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

while True:
    delta=0
    newV=V.copy()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c]=="#":
                continue
            best_value=-100000
            best_action=""
            for action in ACTIONS:
                nr,nc=move((r,c),action)
                value=reward((nr,nc))+GAMMA*V[nr][nc]
                if value>best_value:
                    best_value=value
                    best_action=action
            newV[r][c]=best_value
            policy[r][c]=best_action
            delta=max(delta,abs(best_value-V[r][c]))
    V=newV
    if delta<0.001:
        break

arrow={"Up":"^","Down":"v","Left":"<","Right":">"}

print("\nOptimal Policy\n")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if grid[r][c]=="#":
            print("#",end=" ")
        elif grid[r][c]=="S":
            print("S",end=" ")
        else:
            print(arrow[policy[r][c]],end=" ")
    print()

print("\nRobot Movement\n")

position=start
visited=set()
steps=0
total_reward=0

while steps<20:
    r,c=position
    print("Step",steps," Position:",position," Cell:",grid[r][c])
    if grid[r][c]=="D" and position not in visited:
        total_reward+=DIRT_REWARD
        visited.add(position)
        grid[r][c]="."
        print("Dirt Cleaned")
    action=policy[r][c]
    new_position=move(position,action)
    if new_position==position:
        found=False
        for a in ACTIONS:
            temp=move(position,a)
            if temp!=position:
                new_position=temp
                found=True
                break
        if not found:
            break
    position=new_position
    total_reward+=STEP_COST
    steps+=1

print("\nTotal Reward =",round(total_reward,2))


Output:
Optimal Policy

S > ^ # v 
v # ^ > > 
< < > # ^ 
^ # ^ ^ v 
> ^ < ^ v 

Robot Movement

Step 0  Position: (0, 0)  Cell: S
Step 1  Position: (1, 0)  Cell: .
Step 2  Position: (2, 0)  Cell: D
Dirt Cleaned
Step 3  Position: (1, 0)  Cell: .
Step 4  Position: (2, 0)  Cell: .
Step 5  Position: (1, 0)  Cell: .
Step 6  Position: (2, 0)  Cell: .
Step 7  Position: (1, 0)  Cell: .
Step 8  Position: (2, 0)  Cell: .
Step 9  Position: (1, 0)  Cell: .
Step 10  Position: (2, 0)  Cell: .
Step 11  Position: (1, 0)  Cell: .
Step 12  Position: (2, 0)  Cell: .
Step 13  Position: (1, 0)  Cell: .
Step 14  Position: (2, 0)  Cell: .
Step 15  Position: (1, 0)  Cell: .
Step 16  Position: (2, 0)  Cell: .
Step 17  Position: (1, 0)  Cell: .
Step 18  Position: (2, 0)  Cell: .
Step 19  Position: (1, 0)  Cell: .

Total Reward = 0.2
