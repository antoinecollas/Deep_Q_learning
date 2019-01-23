import torch, sys, copy, time, os
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import SmoothL1Loss
from torch.optim import RMSprop

from deepq.schedule import LinearScheduler
from deepq.utils import eps_greedy_action, init_replay_memory, write_to_tensorboard
from deepq.neural_nets import CNN
from deepq.play import play
from deepq.memory import Memory

def train_deepq(
    name,
    env,
    Q_network,
    input_as_images,
    preprocess_fn=None,
    double_Q=True,
    batch_size=64,
    replay_start_size=50000,
    replay_memory_size=50000,
    agent_history_length=4,
    target_network_update_frequency=10000,
    discount_factor=0.99,
    update_frequency=4,
    eps_scheduler=LinearScheduler(initial_step=1, final_step=0.1, final_timestep=int(1e6)),
    nb_timesteps=int(1e7),
    tensorboard_freq=50,
    ):

    nb_actions = env.action_space.n
    print('NB ACTIONS:', nb_actions)

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

    replay_memory = init_replay_memory(env, agent_history_length, replay_memory_size, replay_start_size, input_as_images, preprocess_fn)

    print('#### TRAINING ####')
    print('see more details on tensorboard')

    done = True #reset environment
    print('Number of trainable parameters:', torch.nn.utils.parameters_to_vector(Q_network.parameters()).shape[0])
    Q_network = Q_network.to(device)
    Q_hat = copy.deepcopy(Q_network).to(device)
    loss = SmoothL1Loss()
    optimizer = RMSprop(Q_network.parameters(), lr=0.00025, momentum=0, alpha=0.95, eps=0.001, centered=True)

    episode = 1
    rewards_episode, total_reward_per_episode, total_gradient_norm = list(), list(), list()

    for timestep in tqdm(range(nb_timesteps)):#tqdm
        #if an episode is ended
        if done:
            #reset the environment
            phi_t = env.reset()
            if preprocess_fn:
                phi_t = preprocess_fn(phi_t)
            last_episodes = Memory(agent_history_length)
            while len(last_episodes.replay_memory)<agent_history_length:
                last_episodes.push(phi_t)

            #tensorboard and save model
            total_reward_per_episode.append(np.sum(rewards_episode))
            rewards_episode = list()
            if (episode%tensorboard_freq == 0):
                assert len(total_reward_per_episode) == tensorboard_freq
                scalars = {
                    'rewards/mean_train_reward': np.mean(total_reward_per_episode),
                    'loss/mean_gradient_norm': np.mean(total_gradient_norm),
                    'other/replay_memory_size': len(replay_memory),
                    'other/eps_exploration': eps_scheduler.get_eps(),
                }
                if input_as_images:
                    demos, demo_rewards = play(env, agent_history_length, Q_network, preprocess_fn, nb_episodes=1, eps=0.01)
                    scalars['rewards/demo_reward'] = np.mean(demo_rewards)
                else:
                    demos = None
                write_to_tensorboard(name, writer, timestep, scalars, Q_network, demos)
                total_reward_per_episode, total_gradient_norm = list(), list()
                
                #save model
                torch.save(Q_network.state_dict(), PATH_SAVE)

            episode += 1

        #choose action
        observations = torch.stack(last_episodes[0:agent_history_length]).to(device)
        if input_as_images:
            observations = observations.unsqueeze(0).permute(0,2,3,1)
        a_t = eps_greedy_action(observations, env, Q_network, eps_scheduler)

        #interact with the environment
        phi_t_1, r_t, done, info = env.step(a_t)

        #for tensorboard
        rewards_episode.append(r_t)

        #preprocess images
        if preprocess_fn:
            phi_t_1 = preprocess_fn(phi_t_1)

        #store in memory
        replay_memory.push([phi_t, a_t, r_t, done])
        phi_t = phi_t_1
        last_episodes.push(phi_t)

        #training
        if timestep % update_frequency == 0:
            #get training data
            phi_t_training, actions_training, r, phi_t_1_training, episode_terminates = replay_memory.sample(batch_size)
            phi_t_training, actions_training, r, phi_t_1_training, episode_terminates = phi_t_training.to(device), actions_training.to(device), r.to(device), phi_t_1_training.to(device), episode_terminates.to(device)
            
            #error
            if double_Q:
                temp = torch.max(Q_network(phi_t_1_training), dim=1)[1]
                Q_hat_values = Q_hat(phi_t_1_training)[torch.arange(temp.shape[0]),temp]
            else:
                Q_hat_values = torch.max(Q_hat(phi_t_1_training), dim=1)[0]

            delta = r + (1-episode_terminates)*discount_factor*Q_hat_values
            delta = delta.detach() #we don't want to compute gradients on target variables

            Q_values = Q_network(phi_t_training)

            mask = torch.zeros([*episode_terminates.shape, nb_actions]).to(device)
            mask.scatter_(1, actions_training.unsqueeze(1), 1.0)
            Q_values = Q_values * mask
            delta = torch.sum(Q_values, dim=1) - delta
            clipped_delta = delta.clamp(-1, 1)
            targets = torch.zeros([*episode_terminates.shape, nb_actions]).to(device)
            targets.masked_scatter_(mask.byte(), delta)

            #backward and gradient descent
            optimizer.zero_grad()
            Q_values.backward(targets.data)
            optimizer.step()

            #tensorboard
            gradient_norm = 0
            for p in Q_network.parameters():
                gradient_norm += torch.sum(p.grad.data**2)
            gradient_norm = np.sqrt(gradient_norm)
            total_gradient_norm.append(gradient_norm)

        if timestep % target_network_update_frequency == 0:
            Q_hat = copy.deepcopy(Q_network).to(device)