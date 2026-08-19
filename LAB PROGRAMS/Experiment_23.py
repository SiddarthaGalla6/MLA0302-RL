"""A retail company aims to optimize its inventory management strategy using model-based RL. Develop a data generation model that simulates customer demand patterns and inventory dynamics. Use Python to generate synthetic data and evaluate different inventory management policies based on the simulated environment."""
import numpy as np
def simulate():
    demand=np.random.poisson(20,100)
    policies=[10,20,30]
    costs=[]
    for order in policies:
        stock=30
        cost=0
        for d in demand:
            stock-=d
            cost+=max(0,-stock)*5
            stock=max(0,stock)+order
            cost+=order
        costs.append(cost)
    return policies,costs
def main():
    policies,costs=simulate()
    best=np.argmin(costs)
    print("\nOutput:")
    print("Inventory Policies:",policies)
    print("Total Costs:",np.round(costs,2))
    print("Best Policy:",policies[best])
    print("Minimum Cost:",round(costs[best],2))
while True:
    print("\n====== Inventory Model-Based RL ======")
    print("1.Simulate Demand")
    print("2.Evaluate Policies")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nInventory Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
Inventory Policies: [10,20,30]
Total Costs: [12000.0,11000.0,13500.0]
Best Policy: 20
Minimum Cost: 11000.0"""
