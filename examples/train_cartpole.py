import gym
from deepq.wrapper_gym import SkipFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import MLP
from deepq.schedule import LinearScheduler

AGENT_HISTORY_LENGTH = 1
OBS_SPACE = 4
env = gym.make("CartPole-v0")
env = SkipFrames(env, AGENT_HISTORY_LENGTH)
Q_network = MLP(OBS_SPACE, env.action_space.n)

train_deepq(
    env=env,
    name='Cartpole',
    Q_network=Q_network,
    agent_history_length=AGENT_HISTORY_LENGTH,
    input_as_images=False,
    tensorboard_freq=500,
    )