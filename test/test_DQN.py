import torch
from dql import DQN

def test_DQN(preprocessed_images):
    batch_size = preprocessed_images.shape[0]
    history_length = preprocessed_images.shape[1]
    nb_actions = 10
    dqn = DQN(history_length, nb_actions)
    q_values = dqn(preprocessed_images)
    assert type(q_values) == torch.Tensor
    assert list(q_values.shape) == [batch_size, nb_actions]