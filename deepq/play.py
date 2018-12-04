# LOAD = False
# if LOAD:
#     list_of_files = glob.glob(DIRECTORY_MODELS)
#     PATH_LOAD = max(list_of_files, key=os.path.getctime) #LOAD lastest created file
    
# if LOAD:
#     print('Load model:', PATH_LOAD)
#     Q.load_state_dict(torch.load(PATH_LOAD))
import copy, torch
import numpy as np
from deepq.utils import get_action

def play(env, Q, preprocess_fn=None, nb_episodes=10, eps=0.1):
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
    env = copy.deepcopy(env)
    episodes, rewards = list(), list()
    for i in range(nb_episodes):
        episode, temp_reward = list(), list()
        observation = env.reset()
        episode.append(observation)
        done = False
        while not done:
            if preprocess_fn:
                phi_t = preprocess_fn(episode[len(episode)-1]).to(device)
            else:
                phi_t = episode[len(episode)-1]
            action = get_action(phi_t, env, Q, eps)
            observation, reward, done, _ = env.step(action)
            episode.append(observation)
            temp_reward.append(reward)
        #we keep only the first frame of each observation
        episode_to_display = list()
        for observation in episode:
            episode_to_display.append(observation[0])
        episodes.append(torch.stack(episode_to_display))
        rewards.append(float(np.sum(temp_reward)))
    
    return episodes, rewards