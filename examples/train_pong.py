import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN
from deepq.utils import preprocess

AGENT_HISTORY_LENGTH = 4
NB_ACTIONS = 6
env = gym.make("PongNoFrameskip-v4")
env = KFrames(env, AGENT_HISTORY_LENGTH)
Q_network = CNN(AGENT_HISTORY_LENGTH, NB_ACTIONS)

train_deepq(
    env=env,
    name='Pong',
    nb_actions=NB_ACTIONS,
    Q_network=Q_network,
    preprocess_fn=preprocess,
    demo_tensorboard=True,
    )