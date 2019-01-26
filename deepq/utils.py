import torch, sys
from tqdm import tqdm
from torchvision.transforms import Compose, ToPILImage, Lambda, Resize, Grayscale, ToTensor
from collections import deque
import random
import numpy as np
from deepq.memory import ExpReplay

def init_replay_memory(env, history_length, replay_memory_size, replay_start_size, input_as_images, preprocess_fn=None, print_info=True):
    '''
    a uniform random policy is run for a number of steps and the resulting experience is used to populate replay memory
    Returns:
    - replay_memory
    '''
    replay_memory = ExpReplay(replay_memory_size, history_length, input_as_images)
    done = True

    if print_info:
        print('#### FILLING REPLAY MEMORY ####')
        range_obj = tqdm(range(replay_start_size))
    else:
        range_obj = range(replay_start_size)

    for i in range_obj:
        #if an episode is ended
        if done:
            phi_t = env.reset()
            if preprocess_fn:
                phi_t = preprocess_fn(phi_t)
        a_t = env.action_space.sample() #random action
        phi_t_1, r_t, done, info = env.step(a_t)
        if preprocess_fn:
            phi_t_1 = preprocess_fn(phi_t_1)
        replay_memory.push([phi_t, a_t, r_t, done])
        phi_t = phi_t_1

    if print_info:
        print('Replay memory is filled with', len(replay_memory), 'transitions. (Max capacity:', replay_memory_size, ').')

    return replay_memory

def preprocess(images):
    ''' 
        Performs preprocessing on a batch of images (bs, h, w, c) or on a single image (h, w, c).
        It doesn't handle flickering!! (there is no flickering in breakout)
        Use grayscale instead of luminance.
    '''
    size_preprocessed_image = 84
    transformations = Compose([
        Lambda(lambda image: image.permute(2,0,1)),
        ToPILImage(),
        Grayscale(),
        Resize((size_preprocessed_image,size_preprocessed_image)),
        ToTensor()
    ])
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    if len(images.shape) == 4:
        batch_size = images.shape[0]
        preprocessed_images = []
        for i in range(batch_size):
            preprocessed_images.append(transformations(images[i]).squeeze(0))
        preprocessed_images = torch.stack(preprocessed_images).permute(1,2,0).squeeze()
    else:
        raise ValueError('tensor s dimension should be 4')    
    return preprocessed_images

def eps_greedy_action(phi_t, env, Q, eps_schedule):
    if type(eps_schedule) is float:
        eps = eps_schedule
    else:
        eps = eps_schedule.step()
    flag_random_exploration = np.random.binomial(n=1, p=eps)

    if flag_random_exploration == 1:
        a_t = env.action_space.sample() #random action
    else:
        device = next(Q.parameters()).device #we assume all parameters are on a same device
        phi_t = phi_t.to(device)
        a_t = torch.argmax(Q(phi_t), dim=1)
    
    return int(a_t)

def write_to_tensorboard(name, writer, episode, scalars, nn, demos=None):
    for key, value in scalars.items():
        writer.add_scalar(key, value, episode)
    if demos:
        #only put first demo in tensorboard
        demo = demos[0].permute([3, 0, 1, 2]).unsqueeze(0)
        writer.add_video(name, demo.numpy().astype(np.uint8), episode, fps=25)
    for name, param in nn.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), episode)