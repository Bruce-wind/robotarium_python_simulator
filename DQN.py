import sys
import os
sys.path.append(os.getcwd() + "/DQN")

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import simulator
from dqn_agent import Agent


env = simulator.Simulator(2, show_figure=False)

agent = Agent(state_size=6, action_size=9, seed=0)

# train a agent
def get_action(index) -> np.ndarray:
    if (index == 0):
        return np.array([-0.2, -3.63])
    elif (index == 1):
        return np.array([-0.2, 0])
    elif (index == 2):
        return np.array([-0.2, 3.63])
    elif (index == 3):
        return np.array([0.0, -3.63])
    elif (index == 4):
        return np.array([0.0, 0])
    elif (index == 5):
        return np.array([0.0, 3.63])
    elif (index == 6):
        return np.array([0.2, -3.63])
    elif (index == 7):
        return np.array([0.2, 0])
    elif (index == 8):
        return np.array([0.2, 3.63])
    else:
        print(f"WARN: invalid index {index}")
        return np.array([0., 0.])

def dqn(n_episodes=100000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
  
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        score = 0
        poses, pose_of_hunter = env.reset(np.array([[2],[2],[0]]))
        state = np.concatenate((poses, pose_of_hunter))
        for t in range(max_t):
            action = agent.act(state, eps)
            real_action = get_action(action) # transform the action to real action
            poses, pose_of_hunter, reward, done= env.step(real_action)
            next_state = np.concatenate((poses, pose_of_hunter)) # concatenate the environment state
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'model\\checkpoint.pth')
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# load the weights from file
# env = simulator.Simulator(2, show_figure=True)
# agent.qnetwork_local.load_state_dict(torch.load('model\\checkpoint.pth'))
# for i in range(3):
#     poses, pose_of_hunter = env.reset(np.array([[2],[2],[0]]))
#     state = np.concatenate((poses, pose_of_hunter))
#     for t in range(3000):
#         action = agent.act(state)
#         real_action = get_action(action) # transform the action to real action
#         poses, pose_of_hunter, reward, done= env.step(real_action)
#         next_state = np.concatenate((poses, pose_of_hunter)) # concatenate the environment state
#         agent.step(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             break
