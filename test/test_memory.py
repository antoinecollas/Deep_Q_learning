import pytest, random, gym, torch
from deepq.memory import ExpReplay
from deepq.utils import init_replay_memory, preprocess

def test_init_replay_memory(env):
    nb_timesteps, env = env
    replay_memory_size = 100
    replay_start_size = 10
    replay_memory = init_replay_memory(env, replay_memory_size=replay_memory_size, replay_start_size=replay_start_size, input_as_images=True, preprocess_fn=preprocess, print_info=False)
    assert len(replay_memory) == replay_start_size
    assert replay_memory[0][0].shape == torch.Size([1, 84, 84, nb_timesteps])
    assert replay_memory[0][1].shape == torch.Size([])
    assert replay_memory[0][2].shape == torch.Size([])
    assert replay_memory[0][3].shape == torch.Size([1, 84, 84, nb_timesteps])
    assert replay_memory[0][4].shape == torch.Size([])
    len_vector = replay_start_size//2
    assert replay_memory[0:len_vector][0].shape == torch.Size([len_vector, 84, 84, nb_timesteps])
    assert replay_memory[0:len_vector][1].shape == torch.Size([len_vector])
    assert replay_memory[0:len_vector][2].shape == torch.Size([len_vector])
    assert replay_memory[0:len_vector][3].shape == torch.Size([len_vector, 84, 84, nb_timesteps])
    assert replay_memory[0:len_vector][4].shape == torch.Size([len_vector])
    
def test_experience_replay(steps_env):
    nb_steps = len(steps_env)
    assert nb_steps >= 3

    size_max_memory = nb_steps + 10
    memory = ExpReplay(size_max_memory)
    assert len(memory) == 0
    for i in range(nb_steps):
        memory.push(steps_env[i])
    assert len(memory) == nb_steps

    #if the memory is too small (nb_steps > size_max_memory)
    size_max_memory = nb_steps - 1
    assert size_max_memory >= 2
    memory = ExpReplay(size_max_memory)
    for i in range(nb_steps):
        memory.push(steps_env[i])
    assert len(memory) == size_max_memory
    #test if the most recents steps are in the memory
    for i in range(len(memory)):
        for j in range(len(memory[i])):
            if (type(memory[len(memory)-1-i][j]) is torch.Tensor) or (type(steps_env[len(steps_env)-1-i][j]) is torch.Tensor):
                assert (memory[len(memory)-1-i][j] == steps_env[len(steps_env)-1-i][j]).all
            else:
                assert memory[len(memory)-1-i][j] == steps_env[len(steps_env)-1-i][j], 'memory['+str(i)+']'+'['+str(j)+']=' + str(memory[len(memory)-1-i][j]) + ' steps_env['+str(i)+']'+'['+str(j)+']=' + str(steps_env[len(steps_env)-1-i][j]) 