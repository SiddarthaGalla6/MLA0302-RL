import numpy as np
PRICES=[10,15,20,25,30]
TRUE_CONV_RATE=[0.60,0.50,0.40,0.25,0.15]
TRUE_REVENUE=[p*r for p,r in zip(PRICES,TRUE_CONV_RATE)]
N_ARMS=len(PRICES)
N_ROUNDS=5000
np.random.seed(1)
def pull_arm(arm):
    purchased=np.random.rand()<TRUE_CONV_RATE[arm]
    return PRICES[arm] if purchased else 0
def epsilon_greedy(epsilon=0.1):
    counts=np.zeros(N_ARMS)
    values=np.zeros(N_ARMS)
    total_revenue=0
    history=[]
    for t in range(N_ROUNDS):
        if np.random.rand()<epsilon:
            arm=np.random.randint(N_ARMS)
        else:
            arm=int(np.argmax(values))
        r=pull_arm(arm)
        counts[arm]+=1
        values[arm]+=(r-values[arm])/counts[arm]
        total_revenue+=r
        history.append(total_revenue)
    return total_revenue,history,counts
def ucb(c=2.0):
    counts=np.zeros(N_ARMS)
    values=np.zeros(N_ARMS)
    total_revenue=0
    history=[]
    for t in range(1,N_ROUNDS+1):
        if t<=N_ARMS:
            arm=t-1
        else:
            ucb_values=values+c*np.sqrt(np.log(t)/counts)
            arm=int(np.argmax(ucb_values))
        r=pull_arm(arm)
        counts[arm]+=1
        values[arm]+=(r-values[arm])/counts[arm]
        total_revenue+=r
        history.append(total_revenue)
    return total_revenue,history,counts
def thompson_sampling():
    alpha=np.ones(N_ARMS)
    beta=np.ones(N_ARMS)
    total_revenue=0
    history=[]
    counts=np.zeros(N_ARMS)
    for t in range(N_ROUNDS):
        samples=np.random.beta(alpha,beta)
        expected_rev=samples*np.array(PRICES)
        arm=int(np.argmax(expected_rev))
        purchased=np.random.rand()<TRUE_CONV_RATE[arm]
        r=PRICES[arm] if purchased else 0
        alpha[arm]+=purchased
        beta[arm]+=(1-purchased)
        counts[arm]+=1
        total_revenue+=r
        history.append(total_revenue)
    return total_revenue,history,counts
if __name__=="__main__":
    print("True Expected Revenue")
    for p,rev in zip(PRICES,TRUE_REVENUE):
        print(f"Price ${p}: {rev:.2f}")
    best_price=PRICES[int(np.argmax(TRUE_REVENUE))]
    print("\nOptimal Price:",best_price)
    eg_rev,eg_hist,eg_counts=epsilon_greedy()
    ucb_rev,ucb_hist,ucb_counts=ucb()
    ts_rev,ts_hist,ts_counts=thompson_sampling()
    print("\nStrategy              Total Revenue   Avg Revenue")
    print("------------------------------------------------")
    print(f"Epsilon-Greedy       {eg_rev:12.2f}   {eg_rev/N_ROUNDS:.4f}")
    print(f"UCB                  {ucb_rev:12.2f}   {ucb_rev/N_ROUNDS:.4f}")
    print(f"Thompson Sampling    {ts_rev:12.2f}   {ts_rev/N_ROUNDS:.4f}")
    print("\nPrice Selection Count")
    print("Price   Epsilon   UCB   Thompson")
    for i,p in enumerate(PRICES):
        print(f"${p:<5}{int(eg_counts[i]):>8}{int(ucb_counts[i]):>7}{int(ts_counts[i]):>11}")
    results={"Epsilon-Greedy":eg_rev,"UCB":ucb_rev,"Thompson Sampling":ts_rev}
    winner=max(results,key=results.get)
    print(f"\nBest Strategy: {winner}")
    print(f"Total Revenue: {results[winner]:.2f}")


Output:
True Expected Revenue
Price $10: 6.00
Price $15: 7.50
Price $20: 8.00
Price $25: 6.25
Price $30: 4.50

Optimal Price: 20

Strategy              Total Revenue   Avg Revenue
------------------------------------------------
Epsilon-Greedy           36280.00   7.2560
UCB                      39940.00   7.9880
Thompson Sampling        39210.00   7.8420

Price Selection Count
Price   Epsilon   UCB   Thompson
$10        317     17        112
$15       4372      1        695
$20        100   4980       3641
$25        107      1        507
$30        104      1         45

Best Strategy: UCB
Total Revenue: 39940.00
