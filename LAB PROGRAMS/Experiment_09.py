'''A call center wants to optimize the assignment of customer service representatives to
incoming calls. Implement a Monte Carlo simulation to estimate the value function for
different assignment policies in Python.'''


from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
CALLS=[]
EPISODES=500
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global CALLS
    CALLS=pd.read_csv(file)
def random_policy():
    rewards=[]
    for _ in range(EPISODES):
        total=0
        for _,r in CALLS.iterrows():
            rep=np.random.randint(1,4)
            if rep==r["BestRep"]:
                total+=10
            else:
                total-=2
        rewards.append(total)
    return np.mean(rewards)
def greedy_policy():
    rewards=[]
    for _ in range(EPISODES):
        total=0
        for _,r in CALLS.iterrows():
            rep=r["BestRep"]
            if rep==r["BestRep"]:
                total+=10
            else:
                total-=2
        rewards.append(total)
    return np.mean(rewards)
def epsilon_policy():
    rewards=[]
    e=0.2
    for _ in range(EPISODES):
        total=0
        for _,r in CALLS.iterrows():
            if np.random.rand()<e:
                rep=np.random.randint(1,4)
            else:
                rep=r["BestRep"]
            if rep==r["BestRep"]:
                total+=10
            else:
                total-=2
        rewards.append(total)
    return np.mean(rewards)
def graph(v):
    names=["Random","Greedy","Epsilon-Greedy"]
    plt.figure(figsize=(6,4))
    plt.bar(names,v,color=["red","green","blue"])
    plt.ylabel("Average Reward")
    plt.title("Policy Comparison")
    plt.grid(axis="y")
    plt.show()
while True:
    print("\n====== Call Center Monte Carlo ======")
    print("1.Upload CSV")
    print("2.Run Monte Carlo Simulation")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
        print("Total Calls :",len(CALLS))
    elif ch=="2":
        r=random_policy()
        g=greedy_policy()
        e=epsilon_policy()
        print("\nEstimated State Value Function")
        print("-------------------------------")
        print("Random Policy         :",round(r,2))
        print("Greedy Policy         :",round(g,2))
        print("Epsilon-Greedy Policy :",round(e,2))
        values=[r,g,e]
        best=np.argmax(values)
        print("\nBest Policy :",["Random","Greedy","Epsilon-Greedy"][best])
        graph(values)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")


Output:
====== Call Center Monte Carlo ======

1.Upload CSV
2.Run Monte Carlo Simulation
3.Exit

Enter Choice: 1

CSV Uploaded Successfully
Total Calls : 15

Enter Choice: 2

Estimated State Value Function
-------------------------------
Random Policy         : 21.74
Greedy Policy         : 150.00
Epsilon-Greedy Policy : 123.62

Best Policy : Greedy
