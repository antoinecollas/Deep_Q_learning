import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import MLP

AGENT_HISTORY_LENGTH = 1
OBS_SPACE = 4
NB_ACTIONS = 2
env = gym.make("CartPole-v0")
env = KFrames(env, AGENT_HISTORY_LENGTH)
Q_network = MLP(OBS_SPACE, NB_ACTIONS)

train_deepq(
    env=env,
    name='Cartpole',
    nb_actions=NB_ACTIONS,
    Q_network=Q_network,
    learning_rate=1e-4,
    tensorboard_freq=500,
    final_exploration_step=int(2*1e6),
    demo_tensorboard=False # not available for cartpole
    )