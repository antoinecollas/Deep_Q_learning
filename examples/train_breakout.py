import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN
from deepq.utils import preprocess

AGENT_HISTORY_LENGTH = 4
NB_ACTIONS = 4
env = gym.make("BreakoutNoFrameskip-v4")
env = KFrames(env, AGENT_HISTORY_LENGTH)
Q_network = CNN(AGENT_HISTORY_LENGTH, NB_ACTIONS)

train_deepq(
    env=env,
    name='Breakout',
    nb_actions=NB_ACTIONS,
    Q_network=Q_network,
    preprocess_fn=preprocess,
    replay_start_size=50000,
    replay_memory_size=50000,
    demo_tensorboard=True,
    )