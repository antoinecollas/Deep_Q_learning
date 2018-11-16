import torch

def test_env(env):
    nb_timesteps, env = env

    #test reset
    observations = env.reset()
    assert type(observations) is torch.Tensor
    assert len(observations.shape) == 4
    assert observations.shape[0] == nb_timesteps
    assert observations.shape[-1] == 3 #nb color channels

    #test step (includind the end of an episode)
    done = False
    while not done:
        a = env.action_space.sample()
        observations, reward, done, _ = env.step(a)
        assert type(observations) is torch.Tensor
        assert len(observations.shape) == 4
        assert observations.shape[0] == nb_timesteps
        assert observations.shape[-1] == 3 #nb color channels
        assert type(reward) is float
        assert type(done) is bool
