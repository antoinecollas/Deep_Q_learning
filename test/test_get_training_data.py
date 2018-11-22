from cnn import CNN
from utils import get_training_data
import torch

def test_get_training_data(replay_memory):
    nb_actions, nb_timesteps, replay_memory = replay_memory
    Q_hat = CNN(agent_history_length=nb_timesteps, nb_actions=nb_actions)
    batch_size = 32
    phi_t_training, actions_training, y = get_training_data(Q_hat, replay_memory, batch_size, 0.99)
    assert type(phi_t_training) == torch.Tensor
    assert phi_t_training.shape[0] == batch_size
    
    assert type(actions_training) == list
    for action in actions_training:
        assert action < nb_actions
        assert action >= 0
    assert len(actions_training) == batch_size
    
    assert type(y) == torch.Tensor
    assert y.shape[0] == batch_size