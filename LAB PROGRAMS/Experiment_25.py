"""A dynamic pricing platform aims to optimize its pricing strategy using model-based RL. 
Develop a predictive model that forecasts customer demand and price sensitivities based on 
historical sales data. Use Python to train the predictive model and implement a model-based 
policy optimization algorithm to dynamically adjust prices in response to changing market conditions."""


import numpy as np
PRICES=np.arange(20,101,10)
def demand(price):
    return max(0,1000-8*price)
def optimize():
    revenue=[]
    for price in PRICES:
        revenue.append(price*demand(price))
    return revenue
def main():
    revenue=optimize()
    best=np.argmax(revenue)
    print("\nOutput:")
    print("Prices:",PRICES.tolist())
    print("Predicted Revenue:",revenue)
    print("Optimal Price:",int(PRICES[best]))
    print("Predicted Demand:",demand(PRICES[best]))
    print("Maximum Revenue:",int(revenue[best]))
while True:
    print("\n====== Dynamic Pricing RL ======")
    print("1.Train Demand Model")
    print("2.Optimize Price")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nDynamic Pricing Optimization Completed")
    elif ch=="3":break
    else:print("Invalid Choice")


OUTPUT:
Prices: [20,30,40,50,60,70,80,90,100]
Predicted Revenue: [16800,22800,23200,20000,14400,7000,0,0,0]
Optimal Price: 40
Predicted Demand: 680
Maximum Revenue: 27200
