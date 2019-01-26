import pytest, gym, torch, random
from deepq.wrapper_gym import SkipFrames
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
    env = SkipFrames(env, skip_frames=nb_timesteps-1)
    return (nb_timesteps, env)

@pytest.fixture('function') #invoked once per test function
def images():
    '''
    Generate batch of images of size: (bs, h, w, c)
    '''
    nb_images = pytest.agent_history_length
    env = gym.make(pytest.env_name)
    phi_t = torch.tensor(env.reset())
    images, done, i = [], False, 1
    images.append(phi_t)
    while (not done) and (i<nb_images):
        a_t = env.action_space.sample() #random action
        phi_t, _, done, _ = env.step(a_t)
        images.append(torch.tensor(phi_t))
        i += 1
    images = torch.stack(images)
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
def replay_memory():
    '''
    Generate a filled replay_memory
    '''
    nb_timesteps = pytest.agent_history_length
    env = gym.make(pytest.env_name)
    nb_actions = env.action_space.n
    env = SkipFrames(env, skip_frames=nb_timesteps-1)
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