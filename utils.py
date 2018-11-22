import torch
from tqdm import tqdm
from torchvision.transforms import Compose, ToPILImage, Lambda, Resize, Grayscale, ToTensor
from collections import deque
import random 
import numpy as np
from memory import Memory

def init_replay_memory(env, replay_memory_size, replay_start_size, print_info=True):
    '''
    a uniform random policy is run for a number of frames and the resulting experience is used to populate replay memory
    Returns:
    - replay_memory
    '''
    replay_memory = Memory(replay_memory_size)
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
            phi_t = preprocess(phi_t)
        a_t = env.action_space.sample() #random action
        phi_t_1, r_t, done, info = env.step(a_t)
        phi_t_1 = preprocess(phi_t_1)
        replay_memory.push([phi_t, a_t, r_t, phi_t_1, done])
        phi_t = phi_t_1

    if print_info:
        print('Replay memory is filled with', len(replay_memory), 'transitions. (Max capacity:', replay_memory_size, ').')

    return replay_memory

def preprocess(images, progress_bar=False):
    ''' 
        Performs preprocessing on a batch of images (bs, h, w, c) or on a single image (h, w, c).
        It doesn't handle flickering!! (there no flickering in breakout)
        Use grayscale instead of luminance.
    '''
    size_preprocessed_image = 84
    transformations = Compose([
        Lambda(lambda image: image.reshape([image.shape[2], image.shape[0], image.shape[1]])),
        ToPILImage(),
        Grayscale(),
        Resize((size_preprocessed_image,size_preprocessed_image)),
        ToTensor()
    ])
    if len(images.shape) == 4:
        batch_size = images.shape[0]
        preprocessed_images = []
        if progress_bar:
            for i in tqdm(range(batch_size)):
                preprocessed_images.append(transformations(images[i]))
        else: 
            for i in range(batch_size):
                preprocessed_images.append(transformations(images[i]))
        preprocessed_images = torch.stack(preprocessed_images).squeeze()
        preprocessed_images = torch.unsqueeze(preprocessed_images, 0)
        if len(preprocessed_images.shape) < 4:
            preprocessed_images = torch.unsqueeze(preprocessed_images, 0)

    else:
        raise ValueError('tensor s dimension should be 4')    
    return preprocessed_images

def action(phi_t, env, Q, eps_schedule):
    eps = eps_schedule.step()
    flag_random_exploration = np.random.binomial(n=1, p=eps)

    if flag_random_exploration == 1:
        a_t = env.action_space.sample() #random action
    else:
        device = next(Q.parameters()).device #we assume all parameters are on a same device
        phi_t = phi_t.to(device)
        a_t = torch.argmax(Q(phi_t), dim=1)
    
    return a_t

def get_training_data(Q_hat, replay_memory, batch_size, discount_factor):
    device = next(Q_hat.parameters()).device #we assume all parameters are on a same device
    y = torch.zeros([batch_size]).to(device)
    transitions_training = replay_memory.sample(batch_size)
    phi_t_training = []
    actions_training = []
    phi_t_1_training = []
    for j in range(len(transitions_training)):
        phi_t_training.append(transitions_training[j][0])
        actions_training.append(transitions_training[j][1])
        phi_t_1_training.append(transitions_training[j][3])

    phi_t_training = torch.squeeze(torch.stack(phi_t_training))
    phi_t_1_training = torch.squeeze(torch.stack(phi_t_1_training)).to(device)
    Q_hat_values = torch.max(Q_hat(phi_t_1_training), dim=1)[0]
    for j in range(len(transitions_training)):
        episode_terminates = transitions_training[j][4]
        if episode_terminates:
            y[j] = transitions_training[j][2]
        else:
            y[j] = transitions_training[j][2] + discount_factor * Q_hat_values[j]
    y = y.detach() #we don't want to compute gradients on target variables
    return phi_t_training, actions_training, y