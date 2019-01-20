import torch, random, time, sys
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

    def __getitem__(self, indices):
        if (type(indices)==int):
            return self.replay_memory[indices]
        elif (type(indices)==slice) or (type(indices)==np.ndarray):
            if (type(indices)==slice):
                start, stop, step = indices.indices(len(self))
                indices = np.arange(start, stop, step, dtype=np.int64)
            to_return = list()
            for i in indices:
                to_return.append(self.replay_memory[i])
            return to_return
        else:
            raise TypeError("index must be int, slice, or numpy array")

    def sample(self, batch_size):
        return random.sample(self.replay_memory, batch_size)
    
    def __len__(self):
        return len(self.replay_memory)

class ExpReplay():
    def __init__(self, replay_memory_size, history_length, images=True):
        self.replay_memory_size = replay_memory_size
        self.filling_level = 0
        self.history_length = history_length
        self.images = images

    def push(self, transition):
        phi_t, a_t, r_t, phi_t_1, done = transition
        if self.images:
            assert len(phi_t.shape) == 2
            assert len(phi_t_1.shape) == 2
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

    def __getitem__(self, last_indices):
        if (type(last_indices)==int):
            phi_t = torch.zeros(1, *(self.phi_t.shape[1:3]), self.history_length)
            phi_t_1 = torch.zeros(1, *(self.phi_t_1.shape[1:3]), self.history_length)

            first_indices = last_indices-(self.history_length-1) if last_indices-(self.history_length-1)>=0 else 0
            list_indices = list(range(last_indices, first_indices))
            while len(list_indices)<self.history_length:
                list_indices.insert(0, 0)
            if self.images:
                phi_t[0] = self.phi_t[list_indices].permute([1,2,0])
                phi_t_1[0] = self.phi_t_1[list_indices].permute([1,2,0])
            else:
                phi_t[0] = self.phi_t[list_indices]
                phi_t_1[0] = self.phi_t_1[list_indices]
            
        elif (type(last_indices)==slice) or (type(last_indices)==np.ndarray):
            if (type(last_indices)==slice):
                start, stop, step = last_indices.indices(len(self))
                last_indices = np.arange(start, stop, step, dtype=np.int64)

            phi_t = torch.zeros(last_indices.shape[0], *(self.phi_t.shape[1:3]), self.history_length).squeeze(-1)
            phi_t_1 = torch.zeros(last_indices.shape[0], *(self.phi_t_1.shape[1:3]), self.history_length).squeeze(-1)

            first_indices = last_indices-(self.history_length-1)*np.ones(last_indices.shape, dtype=np.int64)
            first_indices[first_indices<0] = 0
            for i, (fi, li) in enumerate(zip(first_indices, last_indices)):
                list_indices = list(range(fi, li+1))
                while len(list_indices)<self.history_length:
                    list_indices.insert(0, 0)
                if self.images:
                    phi_t[i] = self.phi_t[list_indices].permute([1,2,0])
                    phi_t_1[i] = self.phi_t_1[list_indices].permute([1,2,0])
                else:
                    phi_t[i] = self.phi_t[list_indices]
                    phi_t_1[i] = self.phi_t_1[list_indices]
        else:
            raise TypeError("index must be int, slice, or numpy array")

        a_t = self.a_t[last_indices]
        r_t = self.r_t[last_indices]
        done = self.done[last_indices]

        return [phi_t, a_t, r_t, phi_t_1, done]
    
    def sample(self, batch_size):
        sample = np.random.randint(self.filling_level, size=batch_size)
        return self[sample]

    def __len__(self):
        return self.filling_level