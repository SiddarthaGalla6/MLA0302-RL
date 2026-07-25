import random
states = ["Low Risk", "Medium Risk", "High Risk"]
actions = ["Allow", "Verify", "Block"]
Q = [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
learning_rate = 0.5
epsilon = 0.3
for episode in range(100):
    state = random.randint(0, 2)
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        action = Q[state].index(max(Q[state]))
    if state == 0 and action == 0:
        reward = 10       
    elif state == 1 and action == 1:
        reward = 10      
    elif state == 2 and action == 2:
        reward = 20       
        reward = -10
    Q[state][action] += learning_rate * (
        reward - Q[state][action]
    )
print("Learned Policy:")
for state in range(3):
    best_action = Q[state].index(max(Q[state]))
    print(states[state], "->", actions[best_action])

Output :
Learned Policy:
Low Risk -> Allow
Medium Risk -> Verify
High Risk -> Block
