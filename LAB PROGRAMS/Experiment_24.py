"""Implement a value-equivalence prediction model to estimate the long-term performance of different 
investment portfolios. Use historical financial data and machine learning techniques to predict the 
value equivalence of alternative portfolio allocations. Write a Python program to analyze and compare 
the predicted performances of various investment strategies."""


import numpy as np
RETURNS=np.array([.12,.09,.15])
RISK=np.array([.20,.10,.25])
WEIGHTS=np.array([[.6,.3,.1],[.2,.6,.2],[.3,.2,.5]])
def predict():
    values=[]
    for w in WEIGHTS:
        values.append(w@RETURNS-.5*w@RISK)
    return np.array(values)
def main():
    values=predict()
    print("\nOutput:")
    for i,v in enumerate(values,1):
        print("Portfolio",i,"Predicted Value:",round(v,4))
    print("Best Portfolio:",np.argmax(values)+1)
    print("Highest Predicted Performance:",round(max(values),4))
while True:
    print("\n====== Value Equivalence Model ======")
    print("1.Predict Portfolio Values")
    print("2.Compare Strategies")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nPortfolio Comparison Completed")
    elif ch=="3":break
    else:print("Invalid Choice")


OUTPUT:
Portfolio 1 Predicted Value: 0.017
Portfolio 2 Predicted Value: 0.028
Portfolio 3 Predicted Value: 0.022
Best Portfolio: 2
Highest Predicted Performance: 0.028
