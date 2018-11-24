# LOAD = False
# if LOAD:
#     list_of_files = glob.glob(DIRECTORY_MODELS)
#     PATH_LOAD = max(list_of_files, key=os.path.getctime) #LOAD lastest created file
    
# if LOAD:
#     print('Load model:', PATH_LOAD)
#     Q.load_state_dict(torch.load(PATH_LOAD))
import copy, torch
import numpy as np
from utils import preprocess, get_action

def play(env, Q, nb_episodes=10, eps=0.1):
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
    episodes = list()
    for i in range(nb_episodes):
        episode = list()
        observation = env.reset()
        episode.append(observation)
        done = False
        while not done:
            phi_t = preprocess(episode[len(episode)-1]).to(device)
            action = get_action(phi_t, env, Q, eps)
            observation, _, done, _ = env.step(action)
            episode.append(observation)
        #we keep only the first frame of each observation
        episode_to_display = []
        for observation in episode:
            episode_to_display.append(observation[0])
        episodes.append(torch.stack(episode_to_display))
    
    return episodes