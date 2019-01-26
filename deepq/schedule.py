import numpy as np

class LinearScheduler():
    '''
    defines a lienar scheduler
    '''
    def __init__(self, steps):
        self.current_idx = -1
        assert steps[0][0] == 0
        x = np.arange(start=0, stop=steps[len(steps)-1][0])
        self.y = np.zeros(shape=steps[len(steps)-1][0])
        for i in range(len(steps)-1):
            a = (steps[i+1][1]-steps[i][1]) / (steps[i+1][0]-steps[i][0])
            b = steps[i][1] - a*steps[i][0]
            self.y[steps[i][0]:steps[i+1][0]] = a*x[steps[i][0]:steps[i+1][0]] + b

    def __getitem__(self, key):
        assert type(key) == int
        if key<len(self.y):
            return self.y[key]
        else:
            return self.y[len(self.y)-1]

    def step(self):
        self.current_idx = self.current_idx+1
        return self[self.current_idx]
    
    def get_eps(self):
        return self[self.current_idx]