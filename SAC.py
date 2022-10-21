import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import torch
import datetime

from SAC.agent import SAC
import simulator
import numpy as np

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class SACConfig:
    def __init__(self) -> None:
        self.algo = 'SAC'
        self.env = 'car'
        # self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = 'sac_models/'  # path to save models
        self.train_eps = 10000
        self.train_steps = 2000
        self.eval_eps = 1
        self.eval_steps = 2000
        self.gamma = 0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2
        self.value_lr  = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 128
        self.batch_size  = 256
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg,seed=1):
    env = simulator.Simulator(2, show_figure=True)
    action_dim = 1  # only control angular velocity
    state_dim  = 6
    agent = SAC(state_dim,action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    def get_real_action(action)->np.ndarray:
        return np.array([0.2, (action[0] + 1) / 2 * 7.26 - 3.63 ])

    for i_ep in range(cfg.train_eps):
        init_pose = np.random.uniform(0.5, 3, size=(3,2))
        state = np.concatenate(env.reset(init_pose))
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            action = agent.policy_net.get_action(state)
            
            action_in = get_real_action(action)
            poses, hunter, reward, done = env.step(action_in)
            next_state = np.concatenate((poses, hunter))

            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward

            # print(f"action: {action}, real_action: {action_in}")
            if done:
                break

        print(f"\rEpisode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}", end="")

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 

        if i_ep % 20 == 0:
            print('\nsave model')
            agent.save(cfg.model_path)
            np.savetxt(cfg.model_path+'reward_{}.txt'.format(curr_time),rewards)
            np.savetxt(cfg.model_path+'ma_reward_{}.txt'.format(curr_time),ma_rewards)

    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward
    agent.eval()

    def get_real_action(action)->np.ndarray:
        return np.array([0.2, (action[0] + 1) / 2 * 7.26 - 3.63 ])
    for i_ep in range(cfg.eval_eps):
        # init_pose = np.random.uniform(-3, 3, size=(3,2))
        init_pose = np.array([[2.5], [2.5], [0]])
        state = np.concatenate(env.reset(init_pose))
        ep_reward = 0
        for i_step in range(cfg.eval_steps):
            action = agent.policy_net.get_action(state)
            
            action_in = get_real_action(action)
            poses, hunter, reward, done = env.step(action_in)
            # poses, hunter, reward, done = env.step(action_in)
            next_state = np.concatenate((poses, hunter))

            # agent.memory.push(state, action, reward, next_state, done)
            # agent.update()
            state = next_state
            ep_reward += reward
            print(f"\rreward: {reward}", end="")
            if done:
                break

        print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
    print('Complete evaling！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg=SACConfig()
    
    # train
    # env,agent = env_agent_config(cfg,seed=1)
    # print("load model")
    # agent.load(path=cfg.model_path)
    # rewards, ma_rewards = train(cfg, env, agent)

    # eval
    env,agent = env_agent_config(cfg,seed=1)
    # agent.load(path='backup/'+cfg.model_path )
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
