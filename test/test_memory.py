import pytest, random, gym, torch, os
from deepq.memory import ExpReplay
from deepq.utils import init_replay_memory, preprocess
import numpy as np
import matplotlib.pyplot as plt

def test_init_replay_memory(env):
    _, env = env
    replay_memory_size = 100
    replay_start_size = 100
    history_length = 4
    first_index = 3
    replay_memory = init_replay_memory(env, history_length=history_length, replay_memory_size=replay_memory_size, replay_start_size=replay_start_size, input_as_images=True, preprocess_fn=preprocess, print_info=False)
    assert len(replay_memory) == replay_start_size
    assert replay_memory[first_index][0].shape == torch.Size([1, 84, 84, history_length])
    assert replay_memory[first_index][1].shape == torch.Size([1])
    assert replay_memory[first_index][2].shape == torch.Size([1])
    assert replay_memory[first_index][4].shape == torch.Size([1])
    len_vector = replay_start_size//2
    assert replay_memory[first_index:len_vector][0].shape == torch.Size([len_vector-first_index, 84, 84, history_length])
    assert replay_memory[first_index:len_vector][1].shape == torch.Size([len_vector-first_index])
    assert replay_memory[first_index:len_vector][2].shape == torch.Size([len_vector-first_index])
    assert replay_memory[first_index:len_vector][4].shape == torch.Size([len_vector-first_index])
    
def test_experience_replay(env):
    _, env = env
    nb_steps = 100
    history_length = 4

    # nb_steps > size_max_memory
    size_max_memory = nb_steps//2
    assert size_max_memory >= 2
    memory = ExpReplay(size_max_memory, history_length)
    observations = []
    done = True
    for i in range(nb_steps):
        if done:
            phi_t = preprocess(env.reset())
            done = False
        a_t = env.action_space.sample() #random action
        phi_t_1, r_t, done, info = env.step(a_t)
        phi_t_1 = preprocess(phi_t_1)
        memory.push([phi_t, a_t, r_t, done])
        observations.append([phi_t, a_t, r_t, done])
        phi_t = phi_t_1

    # visual test
    nb_samples = 20
    samples = memory.sample(batch_size=nb_samples)
    directory = 'visual_tests/test_exp_replay/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(nb_samples):
        for j in range(history_length):
            plt.subplot(2, history_length, j+1)
            plt.imshow(samples[0][i,:,:,j].numpy())
        for j in range(history_length):
            plt.subplot(2, history_length, history_length+j+1)
            plt.imshow(samples[3][i,:,:,j].numpy())

        plt.savefig(directory+'exp_replay_'+str(i)+'.png')