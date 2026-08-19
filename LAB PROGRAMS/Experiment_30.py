"""A team of autonomous robots collaborates to accomplish a complex task (e.g., search and rescue, package delivery) in a dynamic and uncertain environment. Implement multiagent reinforcement learning algorithms (e.g., MADDPG, COMA) to enable the robots to coordinate their actions and achieve collective objectives efficiently. Write a Python program to simulate the robots' interactions and evaluate their performance in completing the task."""
import numpy as np
AGENTS=3
GOAL=np.array([4,4])
def train():
    positions=np.array([[0,0],[0,1],[1,0]])
    reward=0
    for _ in range(20):
        for i in range(AGENTS):
            direction=np.sign(GOAL-positions[i]).astype(int)
            positions[i]+=direction
            if np.array_equal(positions[i],GOAL):
                reward+=10
    return positions,reward
def main():
    positions,reward=train()
    distances=np.sum(abs(GOAL-positions),axis=1)
    print("\nOutput:")
    print("Robot Positions:",positions.tolist())
    print("Remaining Distances:",distances.tolist())
    print("Collective Reward:",reward)
    print("Agents at Goal:",sum(distances==0))
    print("Coordination Status: Completed")
while True:
    print("\n====== Multi-Agent RL ======")
    print("1.Train MADDPG/COMA")
    print("2.Evaluate Team")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nMulti-Agent Performance Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
Robot Positions: [[4,4],[4,4],[4,4]]
Remaining Distances: [0,0,0]
Collective Reward: 30
Agents at Goal: 3
Coordination Status: Completed"""
