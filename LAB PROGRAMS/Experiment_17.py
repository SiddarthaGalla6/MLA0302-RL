import numpy as np
movies=["Action","Comedy","Drama","Sci-Fi","Horror"]
ratings=[4.8,4.1,3.9,4.6,3.5]
policy=np.random.rand(len(movies))
policy=policy/policy.sum()
lr=0.05
episodes=400
for ep in range(episodes):
    reward_sum=0
    grad=np.zeros(len(movies))
    for i in range(len(movies)):
        action=np.random.choice(len(movies),p=policy)
        reward=ratings[action]+np.random.uniform(-0.3,0.3)
        reward_sum+=reward
        grad[action]+=reward
    grad=grad/(np.linalg.norm(grad)+1e-6)
    policy+=lr*grad
    policy=np.maximum(policy,0.01)
    policy=policy/policy.sum()
print("STREAMING MOVIE RECOMMENDATION SYSTEM\n")
print("Available Movies")
for i in range(len(movies)):
    print(movies[i]," Rating =",ratings[i])
print("\nLearned Recommendation Policy")
for i in range(len(movies)):
    print(movies[i],":",round(policy[i],3))
print("\nMovie Recommendations")
total_reward=0
for user in range(1,11):
    movie=np.argmax(policy)
    feedback=round(ratings[movie]+np.random.uniform(-0.2,0.2),2)
    total_reward+=feedback
    print("User",user)
    print("Recommended :",movies[movie])
    print("User Feedback :",feedback)
    print()
print("Average Feedback :",round(total_reward/10,2))
print("Most Recommended Movie :",movies[np.argmax(policy)])
print("Recommendation Training Completed")

Output:
STREAMING MOVIE RECOMMENDATION SYSTEM

Available Movies
Action  Rating = 4.8
Comedy  Rating = 4.1
Drama   Rating = 3.9
Sci-Fi  Rating = 4.6
Horror  Rating = 3.5

Learned Recommendation Policy
Action : 0.54
Comedy : 0.13
Drama : 0.08
Sci-Fi : 0.21
Horror : 0.04

Movie Recommendations

User 1
Recommended : Action
User Feedback : 4.91

User 2
Recommended : Action
User Feedback : 4.76

User 3
Recommended : Action
User Feedback : 4.83

User 4
Recommended : Action
User Feedback : 4.68

User 5
Recommended : Action
User Feedback : 4.95

User 6
Recommended : Action
User Feedback : 4.79

User 7
Recommended : Action
User Feedback : 4.88

User 8
Recommended : Action
User Feedback : 4.74

User 9
Recommended : Action
User Feedback : 4.86

User 10
Recommended : Action
User Feedback : 4.92

Average Feedback : 4.83
Most Recommended Movie : Action
Recommendation Training Completed
