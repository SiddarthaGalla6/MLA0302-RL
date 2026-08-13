'''An online retailer uses a multi-armed bandit approach to set prices dynamically. Simulate
different pricing strategies using epsilon-greedy, UCB, and Thompson Sampling. Write a
Python script to compare which strategy maximizes revenue over a series of pricing
decisions.'''


import numpy as np
import csv
import os
import random
from google.colab import files
ROUNDS=2000
EPSILON=0.1
STATE={'PRICES':[],'PROBS':[],'RESULTS':{}}
def upload_csv():
    print("Please choose a pricing data CSV file to upload...")
    uploaded=files.upload()
    if not uploaded:
        return None
    return list(uploaded.keys())[0]
def download_csv(path):
    files.download(path)
def load_pricing_data(path):
    prices=[]
    probs=[]
    with open(path,'r') as f:
        reader=csv.DictReader(f)
        for row in reader:
            prices.append(float(row['price']))
            probs.append(float(row['conversion_probability']))
    STATE['PRICES']=prices
    STATE['PROBS']=probs
    print("Pricing data loaded successfully from",path)
def epsilon_greedy(prices,probs,rounds,epsilon):
    n=len(prices)
    Q=np.zeros(n)
    N=np.zeros(n)
    total_revenue=0
    for t in range(rounds):
        if random.random()<epsilon:
            arm=random.randint(0,n-1)
        else:
            arm=int(np.argmax(Q))
        convert=1 if random.random()<probs[arm] else 0
        reward=prices[arm]*convert
        N[arm]+=1
        Q[arm]+=(reward-Q[arm])/N[arm]
        total_revenue+=reward
    return total_revenue,N
def ucb(prices,probs,rounds):
    n=len(prices)
    Q=np.zeros(n)
    N=np.zeros(n)
    total_revenue=0
    for t in range(rounds):
        if 0 in N:
            arm=int(np.argmin(N))
        else:
            ucb_values=Q+np.sqrt(2*np.log(t+1)/N)
            arm=int(np.argmax(ucb_values))
        convert=1 if random.random()<probs[arm] else 0
        reward=prices[arm]*convert
        N[arm]+=1
        Q[arm]+=(reward-Q[arm])/N[arm]
        total_revenue+=reward
    return total_revenue,N
def thompson_sampling(prices,probs,rounds):
    n=len(prices)
    alpha=np.ones(n)
    beta_param=np.ones(n)
    N=np.zeros(n)
    total_revenue=0
    for t in range(rounds):
        samples=np.random.beta(alpha,beta_param)
        expected_revenue=samples*np.array(prices)
        arm=int(np.argmax(expected_revenue))
        convert=1 if random.random()<probs[arm] else 0
        if convert:
            alpha[arm]+=1
        else:
            beta_param[arm]+=1
        reward=prices[arm]*convert
        N[arm]+=1
        total_revenue+=reward
    return total_revenue,N
def run_simulation():
    prices=STATE['PRICES']
    probs=STATE['PROBS']
    eg_revenue,eg_counts=epsilon_greedy(prices,probs,ROUNDS,EPSILON)
    ucb_revenue,ucb_counts=ucb(prices,probs,ROUNDS)
    ts_revenue,ts_counts=thompson_sampling(prices,probs,ROUNDS)
    STATE['RESULTS']={'Epsilon-Greedy':(eg_revenue,eg_counts),'UCB':(ucb_revenue,ucb_counts),'Thompson Sampling':(ts_revenue,ts_counts)}
    print("Simulation completed over",ROUNDS,"rounds")
def display_results():
    if not STATE['RESULTS']:
        print("No simulation results available. Please run the simulation first.")
        return
    prices=STATE['PRICES']
    print("Strategy Comparison Results")
    for strategy,(revenue,counts) in STATE['RESULTS'].items():
        print("\nStrategy:",strategy)
        print("Total Revenue:",round(revenue,2))
        for i in range(len(prices)):
            print("  Price",prices[i],"-> Times Selected:",int(counts[i]))
    best_strategy=max(STATE['RESULTS'],key=lambda s:STATE['RESULTS'][s][0])
    print("\nBest Performing Strategy:",best_strategy)
def save_results(path):
    if not STATE['RESULTS']:
        print("No simulation results available. Please run the simulation first.")
        return
    prices=STATE['PRICES']
    with open(path,'w',newline='') as f:
        writer=csv.writer(f)
        header=['Strategy','TotalRevenue']+["Price_"+str(p) for p in prices]
        writer.writerow(header)
        for strategy,(revenue,counts) in STATE['RESULTS'].items():
            row=[strategy,round(revenue,2)]+[int(c) for c in counts]
            writer.writerow(row)
    print("Results saved successfully to",path)
def menu():
    while True:
        print("\n===== Dynamic Pricing Bandit Simulation Menu =====")
        print("1. Upload Pricing Data CSV and Run Simulation")
        print("2. Display Strategy Comparison Results")
        print("3. Save Results to CSV")
        print("4. Exit")
        choice=input("Enter your choice (1-4): ").strip()
        if choice=='1':
            path=upload_csv()
            if path and os.path.exists(path):
                load_pricing_data(path)
                run_simulation()
            else:
                print("No file uploaded. Please try again.")
        elif choice=='2':
            display_results()
        elif choice=='3':
            name=input("Enter output CSV file name (e.g. bandit_results.csv): ").strip()
            if not name:
                name="bandit_results.csv"
            if not name.endswith(".csv"):
                name=name+".csv"
            save_results(name)
            download_csv(name)
        elif choice=='4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
if __name__=="__main__":
    menu()


Output:
===== Dynamic Pricing Bandit Simulation Menu =====
1. Upload Pricing Data CSV and Run Simulation
2. Display Strategy Comparison Results
3. Save Results to CSV
4. Exit
Enter your choice (1-4): 2
Strategy Comparison Results

Strategy: Epsilon-Greedy
Total Revenue: 9880.0
  Price 10.0 -> Times Selected: 1065
  Price 15.0 -> Times Selected: 28
  Price 20.0 -> Times Selected: 41
  Price 25.0 -> Times Selected: 66
  Price 30.0 -> Times Selected: 800

Strategy: UCB
Total Revenue: 9850.0
  Price 10.0 -> Times Selected: 1996
  Price 15.0 -> Times Selected: 1
  Price 20.0 -> Times Selected: 1
  Price 25.0 -> Times Selected: 1
  Price 30.0 -> Times Selected: 1

Strategy: Thompson Sampling
Total Revenue: 11415.0
  Price 10.0 -> Times Selected: 144
  Price 15.0 -> Times Selected: 843
  Price 20.0 -> Times Selected: 753
  Price 25.0 -> Times Selected: 37
  Price 30.0 -> Times Selected: 223

Best Performing Strategy: Thompson Sampling
