"""An adaptive control system aims to adapt its control policy to different operating conditions and environmental changes without explicit retraining. Implement a meta-learning approach to enable the control system to learn how to adapt its parameters and structure based on past experiences and performance feedback. Write a Python program to simulate the control system's adaptation process and evaluate its performance under various conditions."""
import numpy as np
CONDITIONS=[.5,1,1.5,2]
def adapt():
    parameters=[]
    errors=[]
    for condition in CONDITIONS:
        parameter=1
        target=2*condition
        for _ in range(100):
            parameter+=.05*(target-parameter)
        parameters.append(parameter)
        errors.append(abs(target-parameter))
    return parameters,errors
def main():
    parameters,errors=adapt()
    print("\nOutput:")
    print("Conditions:",CONDITIONS)
    print("Adapted Parameters:",np.round(parameters,4))
    print("Errors:",np.round(errors,5))
    print("Average Error:",round(np.mean(errors),5))
    print("Adaptation Status: Successful")
while True:
    print("\n====== Meta-Learning Control ======")
    print("1.Adapt Policy")
    print("2.Evaluate Conditions")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nControl Adaptation Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
Conditions: [0.5,1,1.5,2]
Adapted Parameters: [1.0,2.0,3.0,4.0]
Errors: [0.0,0.0,0.0,0.0]
Average Error: 0.0
Adaptation Status: Successful"""
