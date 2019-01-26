import gym
from deepq.wrapper_gym import SkipFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN2
from deepq.utils import preprocess

AGENT_HISTORY_LENGTH = 4
env = gym.make("PongNoFrameskip-v4")
env = SkipFrames(env, AGENT_HISTORY_LENGTH-1)
Q_network = CNN2(AGENT_HISTORY_LENGTH, env.action_space.n)

train_deepq(
    env=env,
    name='Pong',
    Q_network=Q_network,
    input_as_images=True,
    preprocess_fn=preprocess,
    replay_start_size=5*int(1e4),
    replay_memory_size=int(1e6),
    tensorboard_freq=5,
    )