from google.colab import files
import pandas as pd
import numpy as np
import math
ROWS=0
ADS=[]
def upload_csv():
    f=files.upload()
    return list(f.keys())[0]
def load_csv(file):
    global ADS,ROWS
    df=pd.read_csv(file)
    ADS=df.values
    ROWS=len(df)
def epsilon_greedy():
    d=ADS.shape[1]
    n=[0]*d
    r=[0]*d
    total=0
    eps=0.1
    for i in range(ROWS):
        if np.random.rand()<eps:
            ad=np.random.randint(d)
        else:
            ad=np.argmax(r)
        reward=ADS[i,ad]
        n[ad]+=1
        r[ad]+=((reward-r[ad])/n[ad])
        total+=reward
    return total/ROWS
def ucb():
    d=ADS.shape[1]
    n=[0]*d
    s=[0]*d
    total=0
    for i in range(ROWS):
        ad=0
        m=-1
        for j in range(d):
            if n[j]>0:
                u=s[j]/n[j]+math.sqrt(2*math.log(i+1)/n[j])
            else:
                u=1e9
            if u>m:
                m=u
                ad=j
        reward=ADS[i,ad]
        n[ad]+=1
        s[ad]+=reward
        total+=reward
    return total/ROWS
def thompson():
    d=ADS.shape[1]
    suc=[0]*d
    fail=[0]*d
    total=0
    for i in range(ROWS):
        sample=[np.random.beta(suc[j]+1,fail[j]+1) for j in range(d)]
        ad=np.argmax(sample)
        reward=ADS[i,ad]
        if reward==1:
            suc[ad]+=1
        else:
            fail[ad]+=1
        total+=reward
    return total/ROWS
while True:
    print("\n====== Advertisement Bandit Algorithms ======")
    print("1.Upload CSV")
    print("2.Run Algorithms")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":
        file=upload_csv()
        load_csv(file)
        print("CSV Uploaded Successfully")
        print("Rows :",ROWS)
        print("Advertisements :",ADS.shape[1])
    elif ch=="2":
        eg=epsilon_greedy()
        uc=ucb()
        ts=thompson()
        print("\nClick Through Rate")
        print("--------------------------")
        print("Epsilon Greedy :",round(eg,4))
        print("UCB            :",round(uc,4))
        print("Thompson       :",round(ts,4))
        best=max({"Epsilon Greedy":eg,"UCB":uc,"Thompson Sampling":ts},key={"Epsilon Greedy":eg,"UCB":uc,"Thompson Sampling":ts}.get)
        print("\nBest Algorithm :",best)
    elif ch=="3":
        break
    else:
        print("Invalid Choice")


Output:
====== Advertisement Bandit Algorithms ======
1.Upload CSV
2.Run Algorithms
3.Exit
Enter Choice: 2

Click Through Rate
--------------------------
Epsilon Greedy : 0.4667
UCB            : 0.5333
Thompson       : 0.2667

Best Algorithm : UCB
