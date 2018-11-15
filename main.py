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
BATCH_SIZE = 32
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
NB_EPISODES = 1000

NB_ACTIONS = 4
env = gym.make("BreakoutNoFrameskip-v0")
env = KFrames(env, AGENT_HISTORY_LENGTH)
replay_memory = Memory(REPLAY_MEMORY_SIZE)
done = True

# a uniform random policy is run for a number of frames before learning starts 
# and the resulting experience is used to populate replay memory

print('#### FILLING REPLAY MEMORY ####')
for i in tqdm(range(REPLAY_START_SIZE)):
    #if an episode is ended
    if done:
        phi_t = env.reset()
        phi_t = preprocess(phi_t)
    a_t = env.action_space.sample() #random action
    phi_t_1, r_t, done, info = env.step(a_t)
    phi_t_1 = preprocess(phi_t_1)
    replay_memory.push([phi_t, a_t, r_t, phi_t_1, done])
    phi_t = phi_t_1
print('Replay memory is filled with', len(replay_memory), 'transitions. (Max capacity:', REPLAY_MEMORY_SIZE, ').')

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
            writer.add_scalar('data_per_episode/mean_reward', np.mean(rewards_episode), episode)
            writer.add_scalar('data_per_episode/replay_memory_size', len(replay_memory), episode)
            writer.add_scalar('data_per_episode/eps_exploration', eps, episode)
        phi_t = env.reset()
        phi_t = preprocess(phi_t)
        episode += 1
        rewards_episode = []

    eps = eps_schedule.step()
    flag_random_exploration = np.random.binomial(n=1, p=eps)

    if flag_random_exploration == 1:
        a_t = env.action_space.sample() #random action
    else:
        phi_t = phi_t.to(device)
        a_t = torch.argmax(Q(phi_t), dim=1)

    phi_t_1, r_t, done, info = env.step(a_t)
    rewards_episode.append(r_t)
    phi_t_1 = preprocess(phi_t_1)
    replay_memory.push([phi_t, a_t, r_t, phi_t_1, done])
    phi_t = phi_t_1

    #compute labels (y)
    y = torch.zeros([BATCH_SIZE]).to(device)
    transitions_training = replay_memory.sample(BATCH_SIZE)
    phi_t_training = []
    phi_t_1_training = []
    for j in range(len(transitions_training)):
        phi_t_training.append(transitions_training[j][0])
        phi_t_1_training.append(transitions_training[j][3])

    phi_t_training = torch.squeeze(torch.stack(phi_t_training))
    phi_t_1_training = torch.squeeze(torch.stack(phi_t_1_training)).to(device)
    Q_hat_values = torch.max(Q_hat(phi_t_1_training), dim=1)
    for j in range(len(transitions_training)):
        episode_terminates = transitions_training[4]
        if episode_terminates:
            y[j] = transitions_training[j][2]
        else:
            y[j] = transitions_training[j][2] + DISCOUNT_FACTOR * Q_hat_values[j]

    #forward
    phi_t_training = phi_t_training.to(device)
    Q_values = Q(phi_t_training)
    mask = torch.zeros([BATCH_SIZE, NB_ACTIONS]).to(device)
    for j in range(len(transitions_training)):
        mask[j, transitions_training[j][1]] = 1
    Q_values = Q_values * mask
    Q_values = torch.sum(Q_values, dim=1)
    output = loss(Q_values, y)

    #backward
    optimizer.zero_grad()
    output.backward()

    optimizer.step()

    if (step+1)%TARGET_NETWORK_UPDATE_FREQUENCY == 0:
        print("Update of Q hat!")
        Q_hat = copy.deepcopy(Q).to(device)

    step += 1