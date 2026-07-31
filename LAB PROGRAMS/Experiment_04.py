import numpy as np
ROWS,COLS=6,6
ACTIONS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
GAMMA=0.95
THETA=1e-4
WAREHOUSE=(0,0)
DELIVERY_POINTS={(1,4),(3,2),(5,5)}
OBSTACLES={(2,2),(2,3),(4,4),(0,3)}
def reward(state,target):
    if state==target:
        return 10
    if state in OBSTACLES:
        return -5
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
def policy_evaluation(policy,V,target):
    while True:
        delta=0
        for state in all_states():
            if state==target:
                continue
            a=policy[state]
            ns=next_state(state,a)
            new_v=reward(ns,target)+GAMMA*V[ns]
            delta=max(delta,abs(new_v-V[state]))
            V[state]=new_v
        if delta<THETA:
            break
    return V
def policy_improvement(policy,V,target):
    stable=True
    for state in all_states():
        if state==target:
            continue
        old_action=policy[state]
        action_values={a:reward(next_state(state,a),target)+GAMMA*V[next_state(state,a)] for a in ACTIONS}
        best_action=max(action_values,key=action_values.get)
        policy[state]=best_action
        if best_action!=old_action:
            stable=False
    return policy,stable
def policy_iteration(target):
    V=np.zeros((ROWS,COLS))
    policy={s:np.random.choice(list(ACTIONS)) for s in all_states()}
    iteration=0
    while True:
        V=policy_evaluation(policy,V,target)
        policy,stable=policy_improvement(policy,V,target)
        iteration+=1
        if stable:
            break
    return V,policy,iteration
def find_route(policy,start,target,max_steps=30):
    state=start
    route=[state]
    for _ in range(max_steps):
        if state==target:
            break
        action=policy[state]
        state=next_state(state,action)
        route.append(state)
    return route
def print_grid():
    print("City Grid")
    for r in range(ROWS):
        row=[]
        for c in range(COLS):
            cell=(r,c)
            if cell==WAREHOUSE:
                row.append('W')
            elif cell in DELIVERY_POINTS:
                row.append('P')
            elif cell in OBSTACLES:
                row.append('X')
            else:
                row.append('.')
        print(' '.join(row))
if __name__=="__main__":
    print_grid()
    for target in DELIVERY_POINTS:
        V,policy,iters=policy_iteration(target)
        print(f"\nTarget: {target}")
        print("Iterations:",iters)
        print("Value Function:")
        print(np.round(V,2))
        route=find_route(policy,WAREHOUSE,target)
        print("Route:",route)
        print("Steps:",len(route)-1)


Output:
City Grid
W . . X . .
. . . . P .
. . X X . .
. . P . . .
. . . . X .
. . . . . P

Target: (3, 2)
Iterations: 5
Value Function:
[[ 4.44  5.72  4.44  0.    4.44  3.21]
 [ 5.72  7.07  5.72  4.44  5.72  4.44]
 [ 7.07  8.5   0.    0.    7.07  5.72]
 [ 8.5  10.    0.   10.    8.5   7.07]
 [ 7.07  8.5  10.    8.5   0.    5.72]
 [ 5.72  7.07  8.5   7.07  5.72  4.44]]
Route: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
Steps: 5

Target: (5, 5)
Iterations: 9
Value Function:
[[-1.09 -0.1   0.95  0.    3.21  4.44]
 [-0.1   0.95  2.05  3.21  4.44  5.72]
 [ 0.95  2.05  0.    0.    5.72  7.07]
 [ 2.05  3.21  4.44  5.72  7.07  8.5 ]
 [ 3.21  4.44  5.72  7.07  0.   10.  ]
 [ 4.44  5.72  7.07  8.5  10.    0.  ]]
Route: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
Steps: 10

Target: (1, 4)
Iterations: 7
Value Function:
[[ 4.44  5.72  7.07  0.   10.    8.5 ]
 [ 5.72  7.07  8.5  10.    0.   10.  ]
 [ 4.44  5.72  0.    0.   10.    8.5 ]
 [ 3.21  4.44  5.72  7.07  8.5   7.07]
 [ 2.05  3.21  4.44  5.72  0.    5.72]
 [ 0.95  2.05  3.21  4.44  3.21  4.44]]
Route: [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
Steps: 5
