import gym
from wrapper_gym import KFrames
from deepq import train_deepq

AGENT_HISTORY_LENGTH = 4
env = gym.make("BreakoutNoFrameskip-v4")
env = KFrames(env, AGENT_HISTORY_LENGTH)
NB_ACTIONS = 4

train_deepq(
    env,
    NB_ACTIONS,
    replay_start_size=1000,
    )