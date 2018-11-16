import gym, torch
from gym import Wrapper
import numpy as np
from dql import Memory

class KFrames(Wrapper):
    """
        Implements: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        without maximum component-wise ! (there no flickering in breakout)
    """
    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.skip_frames = history_length - 1

    def reset(self):
        '''
        returns:
        - an array of the observation duplicated history_length times.
        '''
        self.observations = Memory(self.history_length)
        self.rewards = Memory(self.history_length)
        self.done = False
        #we duplicate the first frame in order to have history_length frames
        observation = torch.tensor(self.env.reset())
        for i in range(self.history_length):
            self.observations.push(observation)
        observations = torch.stack(list(self.observations.replay_memory))
        return observations

    def step(self, action):
        '''
        returns:
            - an array of the observations 
            - reward
            - done
            - last info
        '''
        assert not self.done
        j = 0
        sum_rewards = 0.0
        while (not self.done) and (j < self.skip_frames):
            observation, reward, self.done, info = self.env.step(action)
            sum_rewards += reward
            j += 1

        if not self.done:
            observation, reward, self.done, info = self.env.step(action)
            sum_rewards += reward
            self.observations.push(torch.tensor(observation))
            self.rewards.push(sum_rewards)

            total_sum_rewards = 0.0
            for i in range(len(self.rewards)):
                total_sum_rewards += self.rewards[i]
            
        else:
            total_sum_rewards = 0.0

        observations = torch.stack(list(self.observations.replay_memory))
        return observations, total_sum_rewards, self.done, info
    
    def render():
        raise NotImplementedError
        