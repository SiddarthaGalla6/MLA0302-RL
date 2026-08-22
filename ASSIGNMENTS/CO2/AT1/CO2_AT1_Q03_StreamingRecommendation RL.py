'''Question: Illustrate how Reinforcement Learning is used in a streaming platform for content
recommendation. Discuss the impact of delayed rewards, user feedback, and
changing preferences on learning performance.'''

# Code:
import numpy as np
import random
np.random.seed(42)
random.seed(42)
GENRES = ['action', 'comedy', 'drama', 'documentary', 'horror']
N_GENRES = len(GENRES)
MOOD_BINS = ['neutral', 'engaged', 'bored']
N_MOOD = len(MOOD_BINS)
N_STATES = N_GENRES * N_MOOD
ACTIONS = GENRES
N_ACTIONS = len(ACTIONS)
def encode_state(last_genre, mood):
    return last_genre * N_MOOD + mood
def user_preference_profile(week):
    base = np.array([0.3, 0.2, 0.25, 0.1, 0.15])
    if week > 20:
        base = np.array([0.1, 0.35, 0.2, 0.15, 0.2])
    return base / base.sum()
def compute_watch_reward(recommended_genre, mood, preference, delayed_binge):
    match_score = preference[recommended_genre]
    immediate_reward = match_score * 5.0
    if mood == 2:
        immediate_reward -= 3.0
    delayed_reward = 8.0 if delayed_binge else 0.0
    return immediate_reward, delayed_reward
def simulate_user_response(recommended_genre, preference):
    watch_prob = preference[recommended_genre] * 2.0
    watched = random.random() < min(watch_prob, 0.95)
    binge = watched and random.random() < preference[recommended_genre]
    if watched and preference[recommended_genre] > 0.2:
        mood = 1
    elif not watched:
        mood = 2
    else:
        mood = 0
    return watched, binge, mood
def run_recommendation_rl(n_episodes=600):
    Q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma, epsilon = 0.12, 0.85, 0.3
    ep_rewards = []
    for week in range(n_episodes):
        preference = user_preference_profile(week)
        last_genre, mood = random.randint(0, N_GENRES - 1), 0
        total_reward = 0
        pending_delayed = 0.0
        for step in range(10):
            state = encode_state(last_genre, mood)
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                action = int(np.argmax(Q[state]))
            watched, binge, next_mood = simulate_user_response(action, preference)
            immediate_reward, delayed_reward = compute_watch_reward(action, mood, preference, binge)
            reward = immediate_reward + pending_delayed
            pending_delayed = delayed_reward
            next_state = encode_state(action, next_mood)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            last_genre, mood = action, next_mood
            total_reward += reward
        epsilon = max(0.05, epsilon * 0.995)
        ep_rewards.append(total_reward)
    return ep_rewards
print("Content Recommendation RL on a Streaming Platform")
print(f"State Space: last_genre x mood = {N_STATES}")
print(f"Actions (genres to recommend): {ACTIONS}")
rewards = run_recommendation_rl()
print(f"\nAvg reward weeks 1-20 (old preference regime): {np.mean(rewards[:20]):.3f}")
print(f"Avg reward weeks 21-40 (shifting preferences): {np.mean(rewards[20:40]):.3f}")
print(f"Avg reward weeks 41-60 (adapted to new regime): {np.mean(rewards[40:60]):.3f}")
print(f"Avg reward final 50 weeks: {np.mean(rewards[-50:]):.3f}")
print("\nMDP Component Summary:")
print("  States : last watched genre combined with current engagement mood")
print("  Actions:", ACTIONS)
print("  Reward : immediate genre-match score + delayed binge-watch bonus - boredom penalty")
print("\nDelayed Reward and Preference Drift Impact:")
print("  Binge-watch bonus arrives one step after the triggering recommendation, slowing credit assignment")
print("  A mid-simulation preference shift (week 20) temporarily drops reward until the agent re-explores")
print("  Continuous exploration and periodic epsilon reset are needed to track changing user tastes")

'''
Output:
Content Recommendation RL on a Streaming Platform
State Space: last_genre x mood = 15
Actions (genres to recommend): ['action', 'comedy', 'drama', 'documentary', 'horror']

Avg reward weeks 1-20 (old preference regime): 3.663
Avg reward weeks 21-40 (shifting preferences): 3.525
Avg reward weeks 41-60 (adapted to new regime): 8.025
Avg reward final 50 weeks: 21.615

MDP Component Summary:
  States : last watched genre combined with current engagement mood
  Actions: ['action', 'comedy', 'drama', 'documentary', 'horror']
  Reward : immediate genre-match score + delayed binge-watch bonus - boredom penalty

Delayed Reward and Preference Drift Impact:
  Binge-watch bonus arrives one step after the triggering recommendation, slowing credit assignment
  A mid-simulation preference shift (week 20) temporarily drops reward until the agent re-explores
  Continuous exploration and periodic epsilon reset are needed to track changing user tastes
'''
