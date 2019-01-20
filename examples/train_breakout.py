import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN
from deepq.utils import preprocess

AGENT_HISTORY_LENGTH = 4
env = gym.make("BreakoutNoFrameskip-v4")
env = KFrames(env, AGENT_HISTORY_LENGTH-1)
Q_network = CNN(AGENT_HISTORY_LENGTH, env.action_space.n)

train_deepq(
    env=env,
    name='Breakout',
    Q_network=Q_network,
    input_as_images=True,
    preprocess_fn=preprocess,
    replay_start_size=50000,
    replay_memory_size=50000,
    )