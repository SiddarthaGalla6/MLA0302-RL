'''A financial institution wants to optimize its investment strategy. Use a basic policy gradient
method to simulate and optimize the investment policy for maximum returns. Implement
this in Python.'''

                                                     
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
DATA=None
EPISODES=100
LR=0.05
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global DATA
    DATA=pd.read_csv(file)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def policy_gradient():
    theta=0.0
    rewards=[]
    probs=[]
    for ep in range(EPISODES):
        total=0
        p=sigmoid(theta)
        for _,r in DATA.iterrows():
            action=1 if np.random.rand()<p else 0
            ret=r["Return"]
            reward=ret if action==1 else 0
            total+=reward
            grad=(action-p)*reward
            theta+=LR*grad
        rewards.append(total)
        probs.append(sigmoid(theta))
    return rewards,probs
def graph(rewards):
    plt.figure(figsize=(7,4))
    plt.plot(rewards,linewidth=2)
    plt.title("Policy Gradient Learning")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
while True:
    print("\n====== Investment Policy Gradient ======")
    print("1.Upload CSV")
    print("2.Run Policy Gradient")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
        print("Total Records :",len(DATA))
    elif ch=="2":
        rewards,probs=policy_gradient()
        print("\nFinal Policy Probability :",round(probs[-1],4))
        print("Maximum Return :",round(max(rewards),2))
        print("Average Return :",round(np.mean(rewards),2))
        print("Minimum Return :",round(min(rewards),2))
        graph(rewards)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")

Output:
====== Investment Policy Gradient ======

1.Upload CSV
2.Run Policy Gradient
3.Exit

Enter Choice: 1

CSV Uploaded Successfully
Total Records : 15

Enter Choice: 2

Final Policy Probability : 0.9987
Maximum Return : 63
Average Return : 51.28
Minimum Return : 24
