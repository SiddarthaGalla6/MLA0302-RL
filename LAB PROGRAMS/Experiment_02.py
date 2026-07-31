import numpy as np
ROWS,COLS=5,5
ACTIONS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
GAMMA=0.9
THETA=1e-4
ITEM_LOCATIONS={(1,1),(2,3),(3,0)}
GOAL=(4,4)
OBSTACLES={(1,3),(2,2),(3,3)}
def reward(state):
    if state==GOAL:
        return 5
    if state in ITEM_LOCATIONS:
        return 2
    if state in OBSTACLES:
        return -2
    return -0.1
def is_valid(state):
    r,c=state
    return 0<=r<ROWS and 0<=c<COLS
def next_state(state,action):
    dr,dc=ACTIONS[action]
    ns=(state[0]+dr,state[1]+dc)
    return ns if is_valid(ns) else state
def fixed_policy(state):
    r,c=state
    if c<GOAL[1]:
        return 'RIGHT'
    if r<GOAL[0]:
        return 'DOWN'
    return 'RIGHT'
def policy_evaluation(policy_fn):
    V=np.zeros((ROWS,COLS))
    iteration=0
    while True:
        delta=0
        new_V=V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                state=(r,c)
                if state in OBSTACLES or state==GOAL:
                    continue
                action=policy_fn(state)
                ns=next_state(state,action)
                new_V[r,c]=reward(ns)+GAMMA*V[ns]
                delta=max(delta,abs(new_V[r,c]-V[r,c]))
        V=new_V
        iteration+=1
        if delta<THETA:
            break
    return V,iteration
def print_grid():
    print("Warehouse Layout")
    for r in range(ROWS):
        row=[]
        for c in range(COLS):
            cell=(r,c)
            if cell==GOAL:
                row.append('G')
            elif cell in ITEM_LOCATIONS:
                row.append('I')
            elif cell in OBSTACLES:
                row.append('X')
            else:
                row.append('.')
        print(' '.join(row))
if __name__=="__main__":
    print_grid()
    V,iters=policy_evaluation(fixed_policy)
    print("\nIterations:",iters)
    print("\nValue Function:")
    print(np.round(V,2))


Output :
Warehouse Layout
. . . . .
. I . X .
. . X I .
I . . X .
. . . . G

Iterations: 9

Value Function:
[[ 1.87  2.19  2.54  2.94  3.37]
 [ 0.29 -1.9  -2.    0.    3.86]
 [-1.9  -2.    0.    3.86  4.4 ]
 [-1.81 -1.9  -2.    0.    5.  ]
 [ 3.37  3.86  4.4   5.    0.  ]]
