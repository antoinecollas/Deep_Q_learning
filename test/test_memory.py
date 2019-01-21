import pytest, random, gym, torch, os
from deepq.memory import ExpReplay
from deepq.utils import init_replay_memory, preprocess
import numpy as np
import matplotlib.pyplot as plt

def test_init_replay_memory(env):
    history_length, env = env
    replay_memory_size = 100
    replay_start_size = 100
    replay_memory = init_replay_memory(env, history_length=history_length, replay_memory_size=replay_memory_size, replay_start_size=replay_start_size, input_as_images=True, preprocess_fn=preprocess, print_info=False)
    assert len(replay_memory) == replay_start_size
    assert replay_memory[0][0].shape == torch.Size([1, 84, 84, history_length])
    assert replay_memory[0][1].shape == torch.Size([1])
    assert replay_memory[0][2].shape == torch.Size([1])
    assert replay_memory[0][4].shape == torch.Size([1])
    len_vector = replay_start_size//2
    assert replay_memory[0:len_vector][0].shape == torch.Size([len_vector, 84, 84, history_length])
    assert replay_memory[0:len_vector][1].shape == torch.Size([len_vector])
    assert replay_memory[0:len_vector][2].shape == torch.Size([len_vector])
    assert replay_memory[0:len_vector][4].shape == torch.Size([len_vector])
    
def test_experience_replay(env):
    history_length, env = env
    nb_steps = 100

    # nb_steps > size_max_memory
    size_max_memory = nb_steps//2
    assert size_max_memory >= 2
    memory = ExpReplay(size_max_memory, 1)
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

    #test if the most recents steps are in the memory
    for i in range(len(memory)):
        for j in range(len(observations[i])):
            if (type(memory[len(memory)-1-i][j]) is torch.Tensor) or (type(observations[len(observations)-1-i][j]) is torch.Tensor):
                assert (memory[len(memory)-1-i][j] == observations[len(observations)-1-i][j]).all
            else:
                assert memory[len(memory)-1-i][j] == observations[len(observations)-1-i][j], 'memory['+str(i)+']'+'['+str(j)+']=' + str(memory[len(memory)-1-i][j]) + ' observations['+str(i)+']'+'['+str(j)+']=' + str(observations[len(observations)-1-i][j])

    # visual test
    memory.history_length = 4
    directory = 'visual_tests/test_exp_replay/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(min(len(memory), 30)):
        for j in range(memory.history_length):
            plt.subplot(2, memory.history_length, j+1)
            plt.imshow(memory[i][0][0,:,:,j].numpy())
        for j in range(memory.history_length):
            plt.subplot(2, memory.history_length, memory.history_length+j+1)
            plt.imshow(memory[i][3][0,:,:,j].numpy())

        plt.savefig(directory+'exp_replay_'+str(i)+'.png')