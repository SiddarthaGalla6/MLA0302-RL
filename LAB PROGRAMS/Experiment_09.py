import random
agents = ["A1","A2","A3"]
reward = {a:0 for a in agents}
count = {a:0 for a in agents}
for i in range(100):
    agent = random.choice(agents)
    score = random.randint(1,10)
    reward[agent] += score
    count[agent] += 1
print("Call Center Report\n")
for a in agents:
    avg = reward[a] / count[a]
    print(a)
    print("Calls Handled :",count[a])
    print("Average Score :",round(avg,2))
    print()
best = max(agents,key=lambda x:reward[x]/count[x])
print("Best Agent :",best)

Output:
Call Center Report
A1
Calls Handled : 34
Average Score : 6.15
A2
Calls Handled : 33
Average Score : 5.84
A3
Calls Handled : 33
Average Score : 7.09

Best Agent : A3
