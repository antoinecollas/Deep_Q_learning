import random
from play import play
import numpy as np
import matplotlib.pyplot as plt
import imageio

def test_play(env, Q):
    _, env = env
    NB_EPISODES = random.randint(1,5)
    episodes = play(env, Q, NB_EPISODES)
    
    assert type(episodes) is list
    assert len(episodes) == NB_EPISODES
    assert len(episodes[0].shape) == 4 #nb frames, h, w, c
    for episode in episodes:
        assert type(episode) is np.ndarray
        assert episode.shape[-3] == 210 #h
        assert episode.shape[-2] == 160 #w
        assert episode.shape[-1] == 3 #c


    for i, episode in enumerate(episodes):
        imageio.mimwrite('./test_games/game_' + str(i) + '.mp4', episode, fps=25)