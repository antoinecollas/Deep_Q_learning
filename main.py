import gym, torch, sys, copy
from tqdm import tqdm
from dql import *
from wrapper_gym import KFrames
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import SmoothL1Loss
from torch.optim import RMSprop

#GPU/CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#TENSORBOARDX
writer = SummaryWriter()

#HYPERPARAMETERS
BATCH_SIZE = 320
# BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000000
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
DISCOUNT_FACTOR = 0.99
AGENT_HISTORY_LENGTH = ACTION_REPEAT = UPDATE_FREQUENCY = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
# FINAL_EXPLORATION_FRAME = 10000
REPLAY_START_SIZE = 50000
# REPLAY_START_SIZE = 1000
NO_OP_MAX = 30
NB_EPISODES = 100000

NB_ACTIONS = 4
env = gym.make("BreakoutNoFrameskip-v0")
env = KFrames(env, AGENT_HISTORY_LENGTH)

replay_memory = init_replay_memory(env, REPLAY_MEMORY_SIZE, REPLAY_START_SIZE)

print('#### TRAINING ####')
print('see more details on tensorboard')

done = True #reset environment
eps_schedule = ScheduleExploration(INITAL_EXPLORATION, FINAL_EXPLORATION, FINAL_EXPLORATION_FRAME)
Q = DQN(AGENT_HISTORY_LENGTH, NB_ACTIONS).to(device)
Q_hat = copy.deepcopy(Q).to(device)
loss = SmoothL1Loss()
optimizer = RMSprop(Q.parameters(), lr=LEARNING_RATE)

step = 0
episode = 0
rewards_episode = []
while episode < NB_EPISODES:
    #if an episode is ended
    if done:
        #tensorboard
        if len(rewards_episode)>0:
            writer.add_scalar('data_per_episode/reward', np.sum(rewards_episode), episode)
            writer.add_scalar('data_per_episode/replay_memory_size', len(replay_memory), episode)
            writer.add_scalar('data_per_episode/eps_exploration', eps_schedule.get_eps(), episode)
        phi_t = env.reset()
        phi_t = preprocess(phi_t)
        episode += 1
        rewards_episode = []

    a_t = action(phi_t, env, Q, eps_schedule)

    phi_t_1, r_t, done, info = env.step(a_t)
    rewards_episode.append(r_t)
    phi_t_1 = preprocess(phi_t_1)
    replay_memory.push([phi_t, a_t, r_t, phi_t_1, done])
    phi_t = phi_t_1

    #get training data
    phi_t_training, actions_training, y = get_training_data(Q_hat, replay_memory, BATCH_SIZE, DISCOUNT_FACTOR)

    #forward
    phi_t_training = phi_t_training.to(device)
    Q_values = Q(phi_t_training)
    mask = torch.zeros([BATCH_SIZE, NB_ACTIONS]).to(device)
    for j in range(len(actions_training)):
        mask[j, actions_training[j]] = 1
    Q_values = Q_values * mask
    Q_values = torch.sum(Q_values, dim=1)
    output = loss(Q_values, y)

    #backward and gradient descent
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

    if (step+1)%TARGET_NETWORK_UPDATE_FREQUENCY == 0:
        print("Update of Q hat!")
        Q_hat = copy.deepcopy(Q).to(device)

    step += 1