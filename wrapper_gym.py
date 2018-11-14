import gym, torch
from gym import Wrapper
import numpy as np

class KFrames(Wrapper):
    """
        Instead of working frame by frame (like normale gym env), InfinityKFrames works k frames by k frames.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
    
    def reset(self):
        '''
        returns:
        - an array of the observation duplicated k times.
        '''
        #we duplicate the first frame in order to have k frames
        observations = []
        observation = self.env.reset()
        for i in range(self.k):
            observations.append(torch.tensor(observation))
        return torch.stack(observations)

    def step(self, action):
        '''
        returns:
            - an array of the observations 
            - mean reward
            - done
            - last info
        '''
        #becareful shape image tensor
        observations = []
        mean_reward = 0
        done = False
        i = 0
        while (not done) and (i < self.k):
            observation, reward, done, info = self.env.step(action)
            observations.append(torch.tensor(observation))
            mean_reward += reward
            if not done:
                i += 1

        if done == True:
            #if the game is ended, we duplicate the last frame in order to have k frames
            for j in range(i, self.k-1):
                observations.append(observations[i])

        observations = torch.stack(observations)
        mean_reward /= self.k

        return observations, mean_reward, done, info
    
    def render():
        raise NotImplementedError
        