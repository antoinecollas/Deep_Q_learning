import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN
from deepq.utils import preprocess

AGENT_HISTORY_LENGTH = 4
env = gym.make("PongNoFrameskip-v4")
env = KFrames(env, AGENT_HISTORY_LENGTH-1)
Q_network = CNN(AGENT_HISTORY_LENGTH, env.action_space.n)

train_deepq(
    env=env,
    name='Pong',
    Q_network=Q_network,
    input_as_images=True,
    preprocess_fn=preprocess,
    replay_start_size=5*int(1e5),
    replay_memory_size=5*int(1e5),
    tensorboard_freq=5,
    )