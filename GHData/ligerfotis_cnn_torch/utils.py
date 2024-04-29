import numpy as np
from matplotlib import pyplot as plt


def get_args(parser):
    # model type
    parser.add_argument('-m', '--model-size', default='small',
                        help='supports googlenet and vgg16')
    parser.add_argument('-dn', '--dataset-name', default='CIFAR10',
                        help='supports CIFAR10 and MNIST')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print things')
    parser.add_argument('-bs', '--batch-size', default=512, type=int)

    parser.add_argument('-e', '--epochs', default=10, type=int)

    parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float)

    parser.add_argument('-nw', '--num-workers', default=16, type=int)
    parser.add_argument('-wd', '--weight-decay', default=1e-8, type=float)

    parser.add_argument('-dr', '--dropout-rate', default=0.5, type=float)

    args = parser.parse_args()  # running in command line

    return args


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
