import pytest, gym, torch, random
from deepq.wrapper_gym import KFrames
from deepq.utils import preprocess, init_replay_memory
from deepq.neural_nets import CNN

def pytest_namespace():
    return {
        'env_name': 'BreakoutDeterministic-v4',
        'agent_history_length': 4
        # 'agent_history_length': random.randint(1,5)
    }

@pytest.fixture('function') #invoked once per test function
def env():
    '''
    Return (nb_timesteps, wrapper of gym env)
    '''
    nb_timesteps = pytest.agent_history_length
    env = gym.make(pytest.env_name)
    env = KFrames(env, history_length=nb_timesteps)
    return (nb_timesteps, env)

@pytest.fixture('function') #invoked once per test function
def images():
    '''
    Generate batch of images of size: (bs, h, w, c)
    '''
    agent_history_length = pytest.agent_history_length
    images = torch.rand(size=[agent_history_length, 300, 200, 3])
    return images

@pytest.fixture('function') #invoked once per test function
def preprocessed_images():
    '''
    Generate batch of images
    '''
    bs = 5
    images = torch.rand(size=[bs, 84, 84, pytest.agent_history_length])
    return images

@pytest.fixture('function') #invoked once per test function
def steps_env():
    '''
    Generate steps of the environment: (1, timesteps, h, w)
    '''
    nb_timesteps = pytest.agent_history_length
    env = gym.make(pytest.env_name)
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

@pytest.fixture('function') #invoked once per test function
def replay_memory():
    '''
    Generate a filled replay_memory
    '''
    nb_timesteps = pytest.agent_history_length
    env = gym.make(pytest.env_name)
    nb_actions = env.action_space.n
    env = KFrames(env, history_length=nb_timesteps)
    replay_memory = init_replay_memory(env, replay_memory_size=100, replay_start_size=100, input_as_images=True, preprocess_fn=preprocess, print_info=False)
    return nb_actions, nb_timesteps, replay_memory

@pytest.fixture('function') #invoked once per test function
def Q():
    '''
    Generate a Q function
    '''
    agent_history_length = pytest.agent_history_length
    nb_actions = gym.make(pytest.env_name).action_space.n
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Q = CNN(agent_history_length, nb_actions).to(device)
    return Q