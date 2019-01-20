import gym, torch
from gym import Wrapper
import numpy as np
from deepq.memory import Memory

class KFrames(Wrapper):
    """
        Implements: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        without maximum component-wise ! (there no flickering in breakout)
    """
    def __init__(self, env, skip_frames):
        super().__init__(env)
        self.skip_frames = skip_frames

    def reset(self):
        self.done = False
        return torch.FloatTensor(self.env.reset())

    def step(self, action):
        '''
        returns:
            - observation 
            - reward
            - done
            - last info
        '''
        assert not self.done

        sum_rewards = 0
        j = 0
        while (not self.done) and (j < self.skip_frames+1):
            observation, reward, self.done, info = self.env.step(action)
            sum_rewards += reward
            j += 1
        observation = torch.FloatTensor(observation)

        return observation, sum_rewards, self.done, info
    
    def render(self):
        super().render()