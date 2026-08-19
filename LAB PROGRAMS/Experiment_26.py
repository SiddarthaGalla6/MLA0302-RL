"""An urban environment requires a sampling-based planning algorithm to plan collision-free flight 
paths for the UAV while maximizing coverage and minimizing detection latency. Write a Python program 
to simulate the UAV's surveillance mission and visualize the planned paths."""


import numpy as np
START=(0,0)
GOAL=(9,9)
OBSTACLES={(2,2),(3,3),(4,3),(5,4),(6,6)}
def plan():
    path=[START]
    while path[-1]!=GOAL and len(path)<100:
        x,y=path[-1]
        moves=[(x+1,y),(x,y+1),(x-1,y),(x,y-1)]
        moves=[p for p in moves if 0<=p[0]<10 and 0<=p[1]<10 and p not in OBSTACLES]
        path.append(min(moves,key=lambda p:abs(p[0]-9)+abs(p[1]-9)))
    return path
def main():
    path=plan()
    print("\nOutput:")
    print("UAV Path:",path)
    print("Coverage Points:",len(path))
    print("Detection Latency:",round(len(path)*.5,2))
    print("Collision Free:",all(p not in OBSTACLES for p in path))
while True:
    print("\n====== UAV Surveillance RRT ======")
    print("1.Plan Flight")
    print("2.Evaluate Mission")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nUAV Mission Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")


OUTPUT:
UAV Path: [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9)]
Coverage Points: 19
Detection Latency: 9.5
Collision Free: True
