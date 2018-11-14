import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import Compose, ToPILImage, Lambda, Resize, Grayscale, ToTensor
from collections import deque
import random 

def preprocess(images, progress_bar=False):
    ''' 
        Performs preprocessing on a batch of images (bs, h, w, c) or on a single image (h, w, c).
        It doesn't handle flickering!!
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
    else:
        raise ValueError('tensor s dimension should be 4')    
    return preprocessed_images

class Memory():
    def __init__(self, replay_memory_size):
        self.replay_memory = deque()
        self.replay_memory_size = replay_memory_size
    
    def push(self, transition):
        if type(transition) is list:
            for i in range(len(transition)):
                if torch.is_tensor(transition[i]):
                    transition[i] = transition[i].to('cpu')
        if (len(self.replay_memory) >= self.replay_memory_size):
            self.replay_memory.popleft()
        self.replay_memory.append(transition)

    def __getitem__(self, i):
        return self.replay_memory[i]
    
    def sample(self, batch_size):
        return random.sample(self.replay_memory, batch_size)
    
    def __len__(self):
        return len(self.replay_memory)

class ScheduleExploration():
    '''
    defines the exploration schedule (linear in Nature paper)
    '''
    def __init__(self, initial_exploration=1, final_exploration=0.1, final_timestep=1000000/4):
        self.iteration = 0
        self.b = initial_exploration
        self.a = (final_exploration - initial_exploration)/(final_timestep-1)
        self.final_exploration = final_exploration
        self.final_timestep = final_timestep

    def step(self):
        if self.iteration < self.final_timestep:
            res = self.a * self.iteration + self.b
        else:
            res = self.final_exploration
        self.iteration += 1
        return res

class DQN(nn.Module):
    def __init__(self, agent_history_length, nb_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=agent_history_length, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.l1 = nn.Linear(in_features=3136, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=nb_actions)

    def forward(self, x):
       x = self.conv(x)
       x = x.view(x.shape[0], -1)
       x = self.l1(x)
       x = self.l2(x)
       return x