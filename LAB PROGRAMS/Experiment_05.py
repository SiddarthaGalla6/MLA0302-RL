import numpy as np
ROWS,COLS=5,5
ACTIONS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
GAMMA=0.9
THETA=1e-4
PICKUP_POINTS={(0,4):8,(4,0):6,(2,2):10}
OBSTACLES={(1,1),(1,3),(3,1),(3,3)}
TAXI_START=(0,0)
def reward(state):
    if state in PICKUP_POINTS:
        return PICKUP_POINTS[state]
    if state in OBSTACLES:
        return -3
    return -1
def is_valid(state):
    r,c=state
    return 0<=r<ROWS and 0<=c<COLS
def next_state(state,action):
    dr,dc=ACTIONS[action]
    ns=(state[0]+dr,state[1]+dc)
    return ns if is_valid(ns) else state
def all_states():
    return [(r,c) for r in range(ROWS) for c in range(COLS) if (r,c) not in OBSTACLES]
def value_iteration():
    V=np.zeros((ROWS,COLS))
    policy={}
    iteration=0
    while True:
        delta=0
        for state in all_states():
            if state in PICKUP_POINTS:
                continue
            action_values={}
            for a in ACTIONS:
                ns=next_state(state,a)
                action_values[a]=reward(ns)+GAMMA*V[ns]
            best_action=max(action_values,key=action_values.get)
            best_value=action_values[best_action]
            delta=max(delta,abs(best_value-V[state]))
            V[state]=best_value
            policy[state]=best_action
        iteration+=1
        if delta<THETA:
            break
    return V,policy,iteration
def dispatch_route(policy,start,target,max_steps=20):
    state=start
    route=[state]
    for _ in range(max_steps):
        if state==target or state in PICKUP_POINTS:
            break
        state=next_state(state,policy[state])
        route.append(state)
    return route
def print_grid():
    print("City Grid")
    for r in range(ROWS):
        row=[]
        for c in range(COLS):
            cell=(r,c)
            if cell==TAXI_START:
                row.append('T')
            elif cell in PICKUP_POINTS:
                row.append(f'P{PICKUP_POINTS[cell]}')
            elif cell in OBSTACLES:
                row.append('X')
            else:
                row.append('.')
        print(' '.join(row))
if __name__=="__main__":
    print_grid()
    V,policy,iters=value_iteration()
    print("\nIterations:",iters)
    print("\nValue Function:")
    print(np.round(V,2))
    print("\nDispatch Routes:")
    for pickup,rew in PICKUP_POINTS.items():
        route=dispatch_route(policy,TAXI_START,pickup)
        print(f"Pickup {pickup} Reward {rew}")
        print("Route:",route)
        print("Steps:",len(route)-1)


Output :
City Grid
T . . . P8
. X . X .
. . P10 . .
. X . X .
P6 . . . .

Iterations: 5

Value Function:
[[ 4.58  6.2   8.    8.    0.  ]
 [ 6.2   0.   10.    0.    8.  ]
 [ 8.   10.    0.   10.    8.  ]
 [ 6.2   0.   10.    0.    6.2 ]
 [ 0.    6.2   8.    6.2   4.58]]

Dispatch Routes:
Pickup (0, 4) Reward 8
Route: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
Steps: 4
Pickup (4, 0) Reward 6
Route: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
Steps: 4
Pickup (2, 2) Reward 10
Route: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
Steps: 4
