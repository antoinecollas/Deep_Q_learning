import torch
from deepq.utils import preprocess
import matplotlib.pyplot as plt

def test_preprocessing(images):
    preprocessed_images = preprocess(images)

    #shape and type tests
    nb_timesteps = images.shape[0]
    assert type(preprocessed_images) == torch.Tensor
    assert len(preprocessed_images.shape) == 4 #1, h, w, nb_timesteps. One is because ConvNets needs 4 dimensions tensor
    assert preprocessed_images.shape[0] == 1
    assert preprocessed_images.shape[1] == 84
    assert preprocessed_images.shape[2] == 84
    assert preprocessed_images.shape[3] == nb_timesteps

    #visual test: see in 'visual_tests' folder
    plt.subplot(1, 2, 1)
    test_img = plt.imshow(images[0])
    plt.subplot(1, 2, 2)
    test_img = plt.imshow(preprocessed_images[0,:,:,0], cmap='gray')
    plt.savefig('visual_tests/preprocess.png')