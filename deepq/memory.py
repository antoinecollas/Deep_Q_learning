import torch, random, time, sys
from collections import deque
import numpy as np

class Memory():
    def __init__(self, replay_memory_size):
        self.replay_memory = deque(maxlen=replay_memory_size)
    
    def push(self, transition):
        if type(transition) is list:
            for i in range(len(transition)):
                if torch.is_tensor(transition[i]):
                    transition[i] = transition[i].to('cpu')
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
        assert len(transition) == 4
        phi_t, a_t, r_t, done = transition
        if self.images:
            assert len(phi_t.shape) == 2
        if not hasattr(self, 'current_idx'):
            self.current_idx = -1
            shape_memory_imgs = [self.replay_memory_size, *(phi_t.shape)]
            if self.images:
                self.phi_t = torch.zeros(shape_memory_imgs, dtype=torch.uint8)
            else:
                self.phi_t = torch.zeros(shape_memory_imgs)
            self.a_t = torch.zeros(self.replay_memory_size, dtype=torch.long)
            self.r_t = torch.zeros(self.replay_memory_size)
            self.done = torch.zeros(self.replay_memory_size)
        self.current_idx = (self.current_idx+1) % self.replay_memory_size
        if self.filling_level < self.replay_memory_size:
            self.filling_level += 1
        if self.images and phi_t.dtype == torch.float32:
            phi_t = (phi_t.to('cpu') * 255).to(torch.uint8)
            self.phi_t[self.current_idx] = phi_t
        else:
            self.phi_t[self.current_idx] = phi_t.to('cpu')
        self.a_t[self.current_idx] = a_t
        self.r_t[self.current_idx] = r_t
        done = 1 if done else 0
        self.done[self.current_idx] = done

    def __getitem__(self, last_indices):
        if (type(last_indices)==int) or (type(last_indices)==slice) or (type(last_indices)==np.ndarray):
            if (type(last_indices)==int):
                last_indices = np.array([last_indices])
            elif (type(last_indices)==slice):
                start, stop, step = last_indices.indices(len(self))
                last_indices = np.arange(start, stop, step, dtype=np.int64)

            not_available_index_from = self.current_idx
            not_available_index_to = (self.current_idx+self.history_length)%self.filling_level
            if not_available_index_to<=not_available_index_from:
                not_available_index_to += self.filling_level
            not_available_index = np.arange(not_available_index_from, not_available_index_to) % self.filling_level
            if np.sum(np.isin(last_indices, not_available_index))>=1:
                raise TypeError("some indices are not correct")

            phi_t = torch.zeros(last_indices.shape[0], *(self.phi_t.shape[1:3]), self.history_length).squeeze(-1)
            phi_t_1 = torch.zeros(last_indices.shape[0], *(self.phi_t.shape[1:3]), self.history_length).squeeze(-1)

            first_indices = last_indices-(self.history_length-1)*np.ones(last_indices.shape, dtype=np.int64)
            first_indices[first_indices<0] = first_indices[first_indices<0] % self.filling_level

            list_indices = np.zeros((first_indices.shape[0], self.history_length))
            for i, (fi, li) in enumerate(zip(first_indices, last_indices)):
                li = li+1
                if li<=fi:
                    li += self.filling_level
                temp = np.arange(fi, li) % self.filling_level
                list_indices[i, list_indices.shape[1]-temp.shape[0]:] = temp
            list_indices_t_plus_1 = np.roll(list_indices, -1, axis=1)
            list_indices_t_plus_1[:,-1] = (list_indices[:,-1]+1) % self.filling_level

            list_indices, list_indices_t_plus_1 = list_indices.reshape(-1), list_indices_t_plus_1.reshape(-1)
            if self.images:
                phi_t = self.phi_t[list_indices].to(torch.float32) / 255
                phi_t_1 = self.phi_t[list_indices_t_plus_1].to(torch.float32) / 255
                dim1 = int(len(list_indices)/self.history_length)
                dim_image = [self.phi_t[0].shape[0], self.phi_t[0].shape[1]]
                phi_t, phi_t_1 = phi_t.reshape(dim1, self.history_length, *dim_image), phi_t_1.reshape(dim1, self.history_length, *dim_image)
                phi_t, phi_t_1 = phi_t.permute([0,2,3,1]), phi_t_1.permute([0,2,3,1])
            else:
                phi_t = self.phi_t[list_indices].squeeze()
                phi_t_1 = self.phi_t[list_indices_t_plus_1].squeeze()
        else:
            raise TypeError("index must be int, slice, or numpy array")

        a_t = self.a_t[last_indices]
        r_t = self.r_t[last_indices]
        done = self.done[last_indices]

        dim1 = int(len(list_indices)/self.history_length)
        mask = self.done[list_indices].reshape(dim1,self.history_length)
        mask = (-(torch.sum(mask[:,0:mask.shape[1]-1], dim=1, dtype=torch.int64)-1)).nonzero().reshape(-1)
        phi_t, a_t, r_t, phi_t_1, done = phi_t[mask], a_t[mask], r_t[mask], phi_t_1[mask], done[mask]        

        return [phi_t, a_t, r_t, phi_t_1, done]
    
    def sample(self, batch_size):
        available_index_from = self.current_idx+self.history_length
        available_index_to = (self.current_idx-1)
        while available_index_to < available_index_from:
            available_index_to += self.filling_level
        sample = np.random.randint(available_index_from, available_index_to, size=batch_size) % self.filling_level
        to_return = self[sample]
        return to_return

    def __len__(self):
        return self.filling_level