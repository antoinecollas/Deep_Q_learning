import torch
from deepq.utils import preprocess
import matplotlib.pyplot as plt

def test_preprocessing(images):
    #preprocess single image
    preprocessed_image = preprocess(images[0])
    assert type(preprocessed_image) == torch.Tensor
    assert len(preprocessed_image.shape) == 2 #h, w
    assert preprocessed_image.shape[0] == 84
    assert preprocessed_image.shape[1] == 84

    #preprocess batch of images
    preprocessed_images = preprocess(images)
    nb_timesteps = images.shape[0]
    assert type(preprocessed_images) == torch.Tensor
    print(preprocessed_images.shape)
    assert len(preprocessed_images.shape) == 3 #h, w, nb_timesteps
    assert preprocessed_images.shape[0] == 84
    assert preprocessed_images.shape[1] == 84
    assert preprocessed_images.shape[2] == nb_timesteps

    #visual test: see in 'visual_tests' folder
    for i in range(nb_timesteps):
        plt.subplot(nb_timesteps, 2, 2*i+1)
        test_img = plt.imshow(images[i])
        plt.subplot(nb_timesteps, 2, 2*i+2)
        test_img = plt.imshow(preprocessed_images[:,:,i], cmap='gray')
    plt.savefig('visual_tests/preprocess.png')