import torch, time
import numpy as np
import matplotlib.pyplot as plt

def test_env(env):
    nb_timesteps, env = env
    images = []

    #test reset
    observations = env.reset()
    images.append(observations)
    assert type(observations) is torch.Tensor
    assert len(observations.shape) == 4
    assert observations.shape[0] == nb_timesteps
    assert observations.shape[-1] == 3 #nb color channels

    #test step (includind the end of an episode)
    done = False
    while not done:
        a = env.action_space.sample()
        observations, reward, done, _ = env.step(a)
        images.append(observations)
        assert type(observations) is torch.Tensor
        assert len(observations.shape) == 4
        assert observations.shape[0] == nb_timesteps
        assert observations.shape[-1] == 3 #nb color channels
        assert type(reward) is float
        assert type(done) is bool

    #visual test: see in 'visual_tests' folder
    for i, observations in enumerate(images):
        for j, observation in enumerate(observations):
            plt.subplot(1, nb_timesteps, j+1)
            plt.imshow(observation.numpy().astype(np.uint8))
        plt.savefig('visual_tests/env_'+str(i)+'.png')

    #test reward
    env.reset()
    lives = 5
    done = False
    sum_rewards = 0
    _, reward, done, info = env.step(1) # fire
    while not done:
        # env.render()
        _, reward, done, info = env.step(3) # go to the left
        sum_rewards += reward
        # print(sum_rewards)
        # time.sleep(0.2)
        if (not done) and (info['ale.lives'] != lives):
            _, reward, done, info = env.step(1) # fire
            sum_rewards += reward
            # print(sum_rewards)
            # time.sleep(0.2)
            lives -= 1
    #     if done:
    #         time.sleep(1)
    # print(sum_rewards)
    assert sum_rewards == 11 # 11 is the is the score printed in the game (printed with env.render())