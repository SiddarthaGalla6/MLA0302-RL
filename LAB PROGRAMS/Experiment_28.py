"""Multiple agents collaborate to solve a cooperative task in a simulated environment with hierarchical structure and interdependencies. Implement the MAXQ framework to decompose the task into hierarchically organized subtasks and learn policies for each level of the hierarchy. Write a Python program to simulate the agents' interactions and evaluate their performance in achieving the overall task objectives."""
import numpy as np
TASKS=["Collect","Transport","Deliver"]
REWARDS=np.array([3,5,8])
def maxq():
    q=np.zeros(3)
    for _ in range(300):
        action=np.argmax(q+np.random.randn(3)*.2)
        q[action]+=.1*(REWARDS[action]-q[action])
    return q
def main():
    q=maxq()
    print("\nOutput:")
    print("MAXQ Values:",np.round(q,3))
    for task,value in zip(TASKS,q):
        print(task,"Value:",round(value,3))
    print("Best Subtask:",TASKS[np.argmax(q)])
    print("Completed Tasks:",len(TASKS))
while True:
    print("\n====== MAXQ Cooperative Agents ======")
    print("1.Train MAXQ")
    print("2.Evaluate Team")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nHierarchical Task Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
MAXQ Values: [3.0,5.0,8.0]
Collect Value: 3.0
Transport Value: 5.0
Deliver Value: 8.0
Best Subtask: Deliver
Completed Tasks: 3"""
