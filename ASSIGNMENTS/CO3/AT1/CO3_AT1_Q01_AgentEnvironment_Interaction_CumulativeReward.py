'''Question: Predict the Reinforcement Learning (RL) framework by describing the
agent-environment interaction, emphasizing the importance of states, actions, and
rewards, and discuss how cumulative reward influences decision-making.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
N_STATES = 6
N_ACTIONS = 2
GOAL_STATE = 5
def environment_step(state, action):
    if action == 0:
        next_state = max(state - 1, 0)
    else:
        next_state = min(state + 1, N_STATES - 1)
    if next_state == GOAL_STATE:
        reward = 10.0
    elif next_state == state:
        reward = -2.0
    else:
        reward = -0.5
    done = next_state == GOAL_STATE
    return next_state, reward, done
def agent_choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(np.argmax(Q[state]))
Q = np.zeros((N_STATES, N_ACTIONS))
alpha, gamma, epsilon, episodes = 0.15, 0.9, 0.3, 200
episode_returns, episode_steps = [], []
for ep in range(episodes):
    state = 0
    cumulative_reward = 0.0
    steps = 0
    for t in range(30):
        action = agent_choose_action(state, Q, epsilon)
        next_state, reward, done = environment_step(state, action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        cumulative_reward += reward
        state = next_state
        steps += 1
        if done:
            break
    epsilon = max(0.05, epsilon * 0.98)
    episode_returns.append(cumulative_reward)
    episode_steps.append(steps)
print("Reinforcement Learning Framework: Agent-Environment Interaction")
print(f"State Space (agent positions): {N_STATES}")
print(f"Action Space: 0=move_left, 1=move_right")
print(f"Goal State: {GOAL_STATE}")
print(f"\nAvg cumulative reward first 20 episodes: {np.mean(episode_returns[:20]):.3f}")
print(f"Avg cumulative reward last 20 episodes:  {np.mean(episode_returns[-20:]):.3f}")
print(f"Avg steps to goal first 20 episodes:      {np.mean(episode_steps[:20]):.3f}")
print(f"Avg steps to goal last 20 episodes:       {np.mean(episode_steps[-20:]):.3f}")
print("\nAgent-Environment Interaction Cycle:")
print("  1. Agent observes current state from the environment")
print("  2. Agent selects an action using its policy (here, epsilon-greedy over Q)")
print("  3. Environment returns the next state and a reward based on that action")
print("  4. Agent updates its value estimates and the loop repeats until termination")
print("\nRole of Cumulative Reward in Decision-Making:")
print("  The agent does not optimize a single-step reward, it optimizes the sum of rewards over an episode")
print("  Rising cumulative reward across episodes shows the agent learning to reach the goal faster and with fewer penalties")
print("  This return-driven objective is what pushes the policy toward efficient, goal-directed behavior")

'''
Output:
Reinforcement Learning Framework: Agent-Environment Interaction
State Space (agent positions): 6
Action Space: 0=move_left, 1=move_right
Goal State: 5

Avg cumulative reward first 20 episodes: 5.850
Avg cumulative reward last 20 episodes:  7.850
Avg steps to goal first 20 episodes:      8.100
Avg steps to goal last 20 episodes:       5.300

Agent-Environment Interaction Cycle:
  1. Agent observes current state from the environment
  2. Agent selects an action using its policy (here, epsilon-greedy over Q)
  3. Environment returns the next state and a reward based on that action
  4. Agent updates its value estimates and the loop repeats until termination

Role of Cumulative Reward in Decision-Making:
  The agent does not optimize a single-step reward, it optimizes the sum of rewards over an episode
  Rising cumulative reward across episodes shows the agent learning to reach the goal faster and with fewer penalties
  This return-driven objective is what pushes the policy toward efficient, goal-directed behavior
'''
