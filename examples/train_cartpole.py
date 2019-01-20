import gym
from deepq.wrapper_gym import KFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import MLP
from deepq.schedule import LinearScheduler

AGENT_HISTORY_LENGTH = 1
OBS_SPACE = 4
env = gym.make("CartPole-v0")
env = KFrames(env, AGENT_HISTORY_LENGTH)
Q_network = MLP(OBS_SPACE, env.action_space.n)
learning_rate_scheduler = LinearScheduler(initial_step=1e-4, final_step=1e-5, final_timestep=1e6)
eps_scheduler = LinearScheduler(initial_step=1, final_step=0.1, final_timestep=2*1e6)

train_deepq(
    env=env,
    name='Cartpole',
    Q_network=Q_network,
    agent_history_length=AGENT_HISTORY_LENGTH,
    input_as_images=False,
    learning_rate_scheduler=learning_rate_scheduler,
    tensorboard_freq=500,
    eps_scheduler=eps_scheduler,
    )