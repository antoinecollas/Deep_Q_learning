import pytest, gym, torch
from wrapper_gym import KFrames
from dql import preprocess

@pytest.fixture('function') #invoked once per test function
def env():
    '''
    Return (nb_timesteps, wrapper of gym env)
    '''
    nb_timesteps = 4
    env = gym.make("BreakoutNoFrameskip-v0")
    env = KFrames(env, history_length=nb_timesteps)
    return (nb_timesteps, env)

@pytest.fixture('function') #invoked once per test function
def images():
    '''
    Generate batch of images of size: (bs, h, w, c)
    '''
    images = torch.rand(size=[2, 300, 200, 3])
    return images

@pytest.fixture('function') #invoked once per test function
def preprocessed_images():
    '''
    Generate batch of images of size: (1, timesteps, h, w)
    '''
    images = torch.rand(size=[5, 4, 84, 84])
    return images

@pytest.fixture('function') #invoked once per test function
def steps_env():
    '''
    Generate steps of the environment: (1, timesteps, h, w)
    '''
    nb_timesteps = 4
    env = gym.make("BreakoutNoFrameskip-v0")
    env = KFrames(env, history_length=nb_timesteps)
    observations = []
    phi_t = preprocess(env.reset())
    done = False
    i = 0
    while (not done) and (i<10):
        a_t = env.action_space.sample() #random action
        phi_t_1, r_t, done, info = env.step(a_t)
        phi_t_1 = preprocess(phi_t_1)
        observations.append([phi_t, a_t, r_t, phi_t_1, done])
        phi_t = phi_t_1
        i += 1
    return observations