import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    """This function shows the image using

    Arguments:
        img {np array} -- Image numpy array
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
