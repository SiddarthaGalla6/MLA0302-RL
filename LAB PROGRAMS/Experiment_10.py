import numpy as np
policy = 0.5
lr = 0.1
print("Investment Optimization\n")
for i in range(20):
    market = np.random.choice([1,-1], p=[0.6,0.4])
    prob = 1/(1+np.exp(-policy))
    policy += lr * market * (1-prob)
    print("Day",i+1,
          " Market =",market,
          " Policy =",round(policy,3))
print("\nFinal Policy Value :",round(policy,3))
if policy > 0.5:
    print("Decision : Invest More")
else:
    print("Decision : Invest Conservatively")


Output:
Investment Optimization
Day 1  Market = -1  Policy = 0.462
Day 2  Market = 1  Policy = 0.501
Day 3  Market = -1  Policy = 0.463
Day 4  Market = 1  Policy = 0.502
Day 5  Market = -1  Policy = 0.464
Day 6  Market = -1  Policy = 0.425
Day 7  Market = 1  Policy = 0.465
Day 8  Market = 1  Policy = 0.504
Day 9  Market = -1  Policy = 0.466
Day 10  Market = 1  Policy = 0.504
Day 11  Market = 1  Policy = 0.542
Day 12  Market = 1  Policy = 0.579
Day 13  Market = -1  Policy = 0.543
Day 14  Market = -1  Policy = 0.506
Day 15  Market = -1  Policy = 0.469
Day 16  Market = -1  Policy = 0.43
Day 17  Market = -1  Policy = 0.391
Day 18  Market = 1  Policy = 0.431
Day 19  Market = 1  Policy = 0.47
Day 20  Market = 1  Policy = 0.509
Final Policy Value : 0.509
Decision : Invest More
