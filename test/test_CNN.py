import torch
from deepq.neural_nets import CNN

def test_CNN(preprocessed_images):
    batch_size = preprocessed_images.shape[0]
    history_length = preprocessed_images.shape[-1]
    nb_actions = 10
    cnn = CNN(history_length, nb_actions)
    q_values = cnn(preprocessed_images)
    assert type(q_values) == torch.Tensor
    assert list(q_values.shape) == [batch_size, nb_actions]