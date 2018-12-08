import torch, random
from collections import deque
import numpy as np

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

class ExpReplay():
    def __init__(self, replay_memory_size):
        self.replay_memory_size = replay_memory_size
        self.filling_level = 0
    
    def push(self, transition):
        phi_t, a_t, r_t, phi_t_1, done = transition
        phi_t = phi_t.squeeze()
        phi_t_1 = phi_t_1.squeeze()
        if not hasattr(self, 'current_idx'):
            self.current_idx = -1
            shape_memory_imgs = [self.replay_memory_size, *(phi_t.shape)]
            self.phi_t = torch.zeros(shape_memory_imgs)
            self.a_t = torch.zeros(self.replay_memory_size, dtype=torch.long)
            self.r_t = torch.zeros(self.replay_memory_size)
            self.phi_t_1 = torch.zeros(shape_memory_imgs)
            self.done = torch.zeros(self.replay_memory_size)
        self.current_idx = (self.current_idx+1) % self.replay_memory_size
        if self.filling_level < self.replay_memory_size:
            self.filling_level += 1
        self.phi_t[self.current_idx] = phi_t.to('cpu')
        self.a_t[self.current_idx] = a_t
        self.r_t[self.current_idx] = r_t
        self.phi_t_1[self.current_idx] = phi_t_1.to('cpu')
        done = 1 if done else 0
        self.done[self.current_idx] = done

    def __getitem__(self, i):
        phi_t = self.phi_t[i]
        a_t = self.a_t[i]
        r_t = self.r_t[i]
        phi_t_1 = self.phi_t_1[i]
        done = self.done[i]
        return [phi_t, a_t, r_t, phi_t_1, done]
    
    def sample(self, batch_size):
        sample = np.random.randint(self.filling_level, size=batch_size)
        return self[sample]

    def __len__(self):
        return self.filling_level