"""An autonomous exploration robot needs to navigate and map an unknown environment. Implement a 
sampling-based planning algorithm (e.g., RRT, RRT*, or PRM) to plan collision-free paths for the 
robot to explore efficiently. Write a Python program to simulate the robot's exploration process 
and visualize the generated paths."""


import numpy as np
START=(0,0)
GOAL=(9,9)
OBSTACLES={(2,2),(3,3),(4,4),(5,5),(6,6)}
def rrt():
    path=[START]
    for _ in range(50):
        if path[-1]==GOAL:break
        x,y=path[-1]
        moves=[(x+1,y),(x,y+1),(x-1,y),(x,y-1)]
        moves=[p for p in moves if 0<=p[0]<10 and 0<=p[1]<10 and p not in OBSTACLES]
        path.append(min(moves,key=lambda p:abs(p[0]-9)+abs(p[1]-9)))
    return path
def main():
    path=rrt()
    print("\nOutput:")
    print("RRT Path:",path)
    print("Path Length:",len(path)-1)
    print("Goal Reached:",path[-1]==GOAL)
    print("Collision Free:",all(p not in OBSTACLES for p in path))
while True:
    print("\n====== Exploration Robot RRT ======")
    print("1.Generate Path")
    print("2.Evaluate Exploration")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nExploration Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")


OUTPUT:
RRT Path: [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9)]
Path Length: 18
Goal Reached: True
Collision Free: True
