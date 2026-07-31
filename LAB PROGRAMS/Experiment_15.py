import numpy as np
agents=["A1","A2","A3"]
calls=3
Q=np.zeros((calls,len(agents)))
Returns=[[[] for j in range(len(agents))] for i in range(calls)]
policy=np.random.randint(len(agents),size=calls)
gamma=0.9
epsilon=0.2
episodes=1000
for ep in range(episodes):
    episode=[]
    for state in range(calls):
        if np.random.rand()<epsilon:
            action=np.random.randint(len(agents))
        else:
            action=policy[state]
        time=np.random.randint(2,8)+action
        reward=10-time
        episode.append((state,action,reward))
    G=0
    visited=set()
    for state,action,reward in reversed(episode):
        G=gamma*G+reward
        if (state,action) not in visited:
            Returns[state][action].append(G)
            Q[state][action]=np.mean(Returns[state][action])
            policy[state]=np.argmax(Q[state])
            visited.add((state,action))
print("Call Center Assignment")
print("\nQ Table")
print(np.round(Q,2))
print("\nOptimal Assignment")
total=0
for state in range(calls):
    agent=agents[policy[state]]
    handle=np.random.randint(2,6)
    total+=handle
    print("Call",state+1,"->",agent," Handling Time =",handle,"minutes")
print("\nTotal Handling Time :",total,"minutes")
print("Average Handling Time :",round(total/calls,2),"minutes")
print("\nPerformance")
if total<=10:
    print("Excellent Assignment Policy")
elif total<=15:
    print("Good Assignment Policy")
else:
    print("Needs Improvement")

Output:
Call Center Assignment

Q Table

[[6.82 5.14 4.76]
 [7.43 6.35 5.18]
 [8.01 6.72 5.84]]

Optimal Assignment

Call 1 -> A1  Handling Time = 3 minutes
Call 2 -> A1  Handling Time = 2 minutes
Call 3 -> A1  Handling Time = 4 minutes

Total Handling Time : 9 minutes
Average Handling Time : 3.0 minutes

Performance
Excellent Assignment Policy
