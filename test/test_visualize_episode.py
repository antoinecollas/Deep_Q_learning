import random, torch
from deepq.play import play
import matplotlib.pyplot as plt
import imageio
from deepq.utils import preprocess

def test_play(env, Q):
    _, env = env
    NB_EPISODES = random.randint(1,5)
    episodes, rewards = play(env, Q, preprocess, NB_EPISODES)
    
    assert type(episodes) is list
    assert len(episodes) == len(rewards) == NB_EPISODES
    assert len(episodes[0].shape) == 4 #nb frames, h, w, c
    for episode, reward in zip(episodes, rewards):
        assert type(episode) is torch.Tensor
        assert type(reward) is float
        assert episode.shape[-3] == 210 #h
        assert episode.shape[-2] == 160 #w
        assert episode.shape[-1] == 3 #c


    for i, episode in enumerate(episodes):
        imageio.mimwrite('./test_games/game_' + str(i) + '.mp4', episode.numpy(), fps=25)