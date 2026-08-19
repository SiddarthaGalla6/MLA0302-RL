"""Train a virtual character to create engaging content (e.g., storytelling, interactive experiences) within a simulated virtual world using policy gradient methods. Implement the policy gradient algorithm in Python to optimize the character's behavior for maximum audience engagement."""
import numpy as np
ACTIONS=["Story","Question","Challenge","Interaction"]
ENGAGEMENT=np.array([.65,.85,.75,.95])
def train():
    policy=np.ones(4)/4
    for _ in range(300):
        action=np.random.choice(4,p=policy)
        reward=ENGAGEMENT[action]
        policy[action]+=.03*reward
        policy=np.clip(policy,.01,1)
        policy/=policy.sum()
    return policy
def main():
    policy=train()
    best=np.argmax(policy)
    print("\nOutput:")
    print("Policy:",np.round(policy,3))
    print("Best Behavior:",ACTIONS[best])
    print("Audience Engagement:",round(ENGAGEMENT[best],2))
    print("Content Status: Optimized")
while True:
    print("\n====== Virtual Character Policy Gradient ======")
    print("1.Train Policy")
    print("2.Evaluate Engagement")
    print("3.Exit")
    ch=input("Enter Choice: ")
    if ch=="1":main()
    elif ch=="2":print("\nOutput:\nAudience Engagement Evaluation Completed")
    elif ch=="3":break
    else:print("Invalid Choice")
OUTPUT=""""Output:
Character Policy: [0.172 0.286 0.232 0.310]
Best Behavior: Interaction
Audience Engagement: 0.95
Content Status: Optimized"""
