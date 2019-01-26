import copy, torch
import numpy as np
from deepq.memory import Memory
from deepq.utils import eps_greedy_action, preprocess
from deepq.wrapper_gym import SkipFrames
import gym

def play_atari(env_name, agent_history_length, Q, nb_episodes=10, eps=0.1):
    '''
    Input:
        - environment (the environment is copied, so it is not modified)
        - Q function
        - number of episodes to play
        - eps between 0 and 1. It adds some randomness.
    Returns:
        - list of observations (np arrays, one np array == one episode played)
    '''
    device = next(Q.parameters()).device
    env = gym.make(env_name)
    env = SkipFrames(env, agent_history_length-1)
    episodes, rewards = list(), list()
    for i in range(nb_episodes):
        episode, temp_reward = list(), list()
        observation = env.reset()
        episode.append(observation)
        done = False
        memory = Memory(agent_history_length)
        while not done:
            phi_t = preprocess(episode[len(episode)-1]).to(device)
            memory.push(phi_t)
            while len(memory.replay_memory)<agent_history_length:
                memory.push(phi_t)
            phi_t = torch.stack(memory[0:agent_history_length]).unsqueeze(0).to(device)
            phi_t = phi_t.permute(0,2,3,1)
            action = eps_greedy_action(phi_t, env, Q, eps)
            observation, reward, done, _ = env.step(action)
            episode.append(observation)
            temp_reward.append(reward)
        episodes.append(torch.stack(episode))
        rewards.append(float(np.sum(temp_reward)))
    
    return episodes, rewards