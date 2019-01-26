import gym, torch
from gym import Wrapper
import numpy as np
from deepq.memory import Memory

class SkipFrames(Wrapper):
    """
        Implements: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    """
    def __init__(self, env, skip_frames, preprocess_fn=None):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.preprocess_fn = preprocess_fn
        if self.preprocess_fn:
            assert skip_frames>0 #because we have to keep 2 frames to prevent flickering

    def reset(self):
        self.done = False
        oberservation = torch.FloatTensor(self.env.reset())
        if self.preprocess_fn:
            oberservation = self.preprocess_fn(oberservation)
        return oberservation

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
        if self.preprocess_fn:
            observations = []
        while (not self.done) and (j < self.skip_frames+1):
            if self.preprocess_fn:
                observation, reward, self.done, info = self.env.step(action)
                observations.append(torch.FloatTensor(observation))
            else:
                observation, reward, self.done, info = self.env.step(action)
            sum_rewards += reward
            j += 1
        
        if self.preprocess_fn:
            observation = self.preprocess_fn(torch.stack(observations)[-2:])
        observation = torch.FloatTensor(observation)

        return observation, sum_rewards, self.done, info
    
    def render(self):
        super().render()