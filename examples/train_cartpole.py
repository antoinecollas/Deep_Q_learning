import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import MLP

AGENT_HISTORY_LENGTH = 1
OBS_SPACE = 4
env = gym.make("CartPole-v0")
env = KFrames(env, AGENT_HISTORY_LENGTH)
Q_network = MLP(OBS_SPACE, env.action_space.n)

train_deepq(
    env=env,
    name='Cartpole',
    Q_network=Q_network,
    input_images=False,
    learning_rate=1e-4,
    tensorboard_freq=500,
    final_exploration_step=int(2*1e6),
    )