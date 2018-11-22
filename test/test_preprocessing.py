import torch
from utils import preprocess

def test_preprocessing(images):
    batch_size = images.shape[0]
    preprocessed_images = preprocess(images)
    assert type(preprocessed_images) == torch.Tensor
    assert len(preprocessed_images.shape) == 4 #1, bs, h, w. One is because ConvNets needs 4 dimensions tensor
    assert list(preprocessed_images.shape) == [1, batch_size, 84, 84]