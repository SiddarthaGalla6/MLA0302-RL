Question: Sampled trajectories generation and evaluation
import random
random.seed(12)
stages=["Environment","State","Model","Planning","Action","Reward","Policy"]
print("CO4 AT2 - Question 12: Sampled trajectories generation and evaluation")
print("\nWorkflow / Concept Diagram")
for j in range(len(stages)-1):
    print(stages[j]+"  ->  "+stages[j+1])
print("\nLearning Iterations")
scores=[]
for step in range(1,7):
    state=random.randint(1,10)
    action=random.choice(["A1","A2","A3"])
    reward=random.randint(-2,10)
    prediction=state+reward
    gradient=round((reward-prediction)*0.01,3)
    score=round(prediction-gradient,3)
    scores.append(score)
    print(f"Step {step}: State={state} Action={action} Reward={reward} Prediction={prediction} Gradient={gradient}")
print("\nComponent Roles")
print("Environment: provides states, transitions and rewards")
print("Model: predicts future states and reward outcomes")
print("Planning: evaluates possible actions or trajectories")
print("Sampling: explores representative candidate outcomes")
print("Value: estimates expected future return")
print("Gradient: indicates how parameters should change")
print("Optimizer: updates parameters to improve performance")
print("Policy: selects actions with higher expected return")
print("\nPerformance Summary")
print("Average Learning Score:",round(sum(scores)/len(scores),2))
print("Iterations:",len(scores))
print("Decision Rule: choose the action with best predicted return")
print("Feedback Loop: action -> reward -> learning -> improved policy")
print("Final Result: improved decision making through repeated optimization")
print("\nConcept Flow")
print("Experience -> Prediction -> Evaluation -> Optimization -> Better Policy")
print("Model knowledge reduces unnecessary environment interaction")
print("Rewards provide the objective signal for improvement")
print("The process repeats until performance becomes stable")
