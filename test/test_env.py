import torch, time, os
import numpy as np
import matplotlib.pyplot as plt

def test_env(env):
    _, env = env
    images = []

    #test reset
    observations = env.reset()
    images.append(observations)
    assert type(observations) is torch.Tensor
    assert len(observations.shape) == 3
    assert observations.shape[-1] == 3 #nb color channels

    #test step (includind the end of an episode)
    done = False
    while not done:
        a = env.action_space.sample()
        observations, reward, done, _ = env.step(a)
        images.append(observations)
        assert type(observations) is torch.Tensor
        assert len(observations.shape) == 3
        assert observations.shape[-1] == 3 #nb color channels
        assert type(reward) is float
        assert type(done) is bool

    #visual test: see in 'visual_tests' folder
    directory = 'visual_tests/test_env/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, observation in enumerate(images):
        plt.imshow(observation.numpy().astype(np.uint8))
        plt.savefig(directory+'env_'+str(i)+'.png')

    #test reward
    env.reset()
    lives = 5
    done = False
    sum_rewards = 0
    _, reward, done, info = env.step(1) # fire
    while not done:
        _, reward, done, info = env.step(3) # go to the left
        sum_rewards += reward
        if (not done) and (info['ale.lives'] != lives):
            _, reward, done, info = env.step(1) # fire
            sum_rewards += reward
            lives -= 1

    assert sum_rewards == 11 # 11 is the is the score printed in the game (printed with env.render())