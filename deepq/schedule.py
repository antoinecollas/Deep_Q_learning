class LinearScheduler():
    '''
    defines a lienar scheduler
    '''
    def __init__(self, initial_step=1, final_step=0.1, final_timestep=1e6):
        self.iteration = 0
        self.b = initial_step
        self.a = (final_step - initial_step)/(final_timestep-1)
        self.final_step = final_step
        self.final_timestep = final_timestep
        self.eps = initial_step

    def step(self):
        if self.iteration < self.final_timestep:
            res = self.a * self.iteration + self.b
        else:
            res = self.final_step
        self.iteration += 1
        self.eps = res
        return res
    
    def get_eps(self):
        return float(self.eps)