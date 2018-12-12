import torch
from deepq.utils import preprocess

def test_preprocessing(images):
    nb_timesteps = images.shape[0]
    preprocessed_images = preprocess(images)
    assert type(preprocessed_images) == torch.Tensor
    assert len(preprocessed_images.shape) == 4 #1, h, w, nb_timesteps. One is because ConvNets needs 4 dimensions tensor
    assert list(preprocessed_images.shape) == [1, 84, 84, nb_timesteps]