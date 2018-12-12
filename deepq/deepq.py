import torch, sys, copy, time, os
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import SmoothL1Loss
from torch.optim import RMSprop

from deepq.wrapper_gym import KFrames
from deepq.schedule import ScheduleExploration
from deepq.utils import eps_greedy_action, get_training_data, init_replay_memory, write_to_tensorboard
from deepq.neural_nets import CNN
from deepq.play import play

def train_deepq(
    name,
    env,
    Q_network,
    input_images,
    preprocess_fn=None,
    batch_size=32,
    replay_start_size=50000,
    replay_memory_size=50000,
    agent_history_length=4,
    target_network_update_frequency=10000,
    discount_factor=0.99,
    learning_rate=1e-5,
    update_frequency=4,
    inital_exploration=1,
    final_exploration=0.1,
    final_exploration_step=int(1e6),
    nb_timesteps=int(1e7),
    tensorboard_freq=50,
    ):

    nb_actions = env.action_space.n

    #SAVE/LOAD MODEL
    DIRECTORY_MODELS = './models/'
    if not os.path.exists(DIRECTORY_MODELS):
        os.makedirs(DIRECTORY_MODELS)
    PATH_SAVE = DIRECTORY_MODELS + name + '_' + time.strftime('%Y%m%d-%H%M')

    #GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('RUNNING ON', device)

    #TENSORBOARDX
    writer = SummaryWriter(comment=name)

    replay_memory = init_replay_memory(env, replay_memory_size, replay_start_size, input_images, preprocess_fn)

    print('#### TRAINING ####')
    print('see more details on tensorboard')

    done = True #reset environment
    eps_schedule = ScheduleExploration(inital_exploration, final_exploration, final_exploration_step)
    Q_network = Q_network.to(device)
    print('Number of trainable parameters:', Q_network.count_parameters())
    Q_hat = copy.deepcopy(Q_network).to(device)
    loss = SmoothL1Loss()
    optimizer = RMSprop(Q_network.parameters(), lr=learning_rate, alpha=0.95, eps=0.01, centered=True)

    episode = 1
    rewards_episode, total_reward_per_episode, total_loss = list(), list(), list()
    for timestep in tqdm(range(nb_timesteps)):#tqdm
        #if an episode is ended
        if done:
            phi_t = env.reset()
            if preprocess_fn:
                phi_t = preprocess_fn(phi_t)

            total_reward_per_episode.append(np.sum(rewards_episode))
            rewards_episode = list()
            if (episode%tensorboard_freq == 0):
                assert len(total_reward_per_episode) == tensorboard_freq
                scalars = {
                    'rewards/train_reward': np.mean(total_reward_per_episode),
                    'loss/train_loss': np.mean(total_loss),
                    'other/replay_memory_size': len(replay_memory),
                    'other/eps_exploration': eps_schedule.get_eps()
                }
                if input_images:
                    demos, demo_rewards = play(env, Q_network, preprocess_fn, nb_episodes=1, eps=eps_schedule.get_eps())
                    scalars['rewards/demo_reward'] = np.mean(demo_rewards)
                else:
                    demos = None
                write_to_tensorboard(name, writer, episode, scalars, demos)
                total_reward_per_episode, total_loss = list(), list()
                
                #save model
                torch.save(Q_network.state_dict(), PATH_SAVE)

            episode += 1

        a_t = eps_greedy_action(phi_t, env, Q_network, eps_schedule)

        phi_t_1, r_t, done, info = env.step(a_t)
        rewards_episode.append(r_t)
        if preprocess_fn:
            phi_t_1 = preprocess_fn(phi_t_1)
        replay_memory.push([phi_t, a_t, r_t, phi_t_1, done])
        phi_t = phi_t_1

        #training
        if timestep % update_frequency == 0:
            #get training data
            phi_t_training, actions_training, y = get_training_data(Q_hat, replay_memory, batch_size, discount_factor)

            #forward
            phi_t_training = phi_t_training.to(device)
            Q_values = Q_network(phi_t_training)
            mask = torch.zeros([batch_size, nb_actions]).to(device)
            mask.scatter_(1, actions_training.unsqueeze(1), 1.0)
            Q_values = Q_values * mask
            Q_values = torch.sum(Q_values, dim=1)
            output = loss(Q_values, y)
            total_loss.append(float(output))

            #backward and gradient descent
            optimizer.zero_grad()
            output.backward()
            optimizer.step()

        if timestep % target_network_update_frequency == 0:
            Q_hat = copy.deepcopy(Q_network).to(device)