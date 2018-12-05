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
        self.eps = initial_exploration

    def step(self):
        if self.iteration < self.final_timestep:
            res = self.a * self.iteration + self.b
        else:
            res = self.final_exploration
        self.iteration += 1
        self.eps = res
        return res
    
    def get_eps(self):
        return float(self.eps)