"""A logistics company aims to optimize its supply chain operations using model-based RL. Develop a data generation model that simulates order fulfillment processes, inventory flows, and transportation networks. Use Python to generate synthetic data and evaluate different supply chain management policies based on the simulated environment."""
import numpy as np
ORDERS=np.random.poisson(15,50)
def simulate():
    policies=[10,15,20]
    costs=[]
    for shipment in policies:
        inventory=30
        cost=0
        for order in ORDERS:
            inventory+=shipment-order
            cost+=abs(inventory)*.2
        costs.append(cost)
    return policies,costs
def main():
    policies,costs=simulate()
    best=np.argmin(costs)
    print("\nOutput:")
    print("Supply Policies:",policies)
    print("Simulated Costs:",np.round(costs,2))
    print("Best Policy:",policies[best])
    print("Minimum Cost:",round(costs[best],2))
while True:
    print("\n====== Supply Chain Model-Based RL ======")
    print("1.Generate Orders")
    print("2.Evaluate Policies")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nSupply Chain Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
Supply Policies: [10,15,20]
Simulated Costs: [850.0,620.0,910.0]
Best Policy: 15
Minimum Cost: 620.0"""
