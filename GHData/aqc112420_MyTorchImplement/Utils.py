import os
from scipy.misc import imread
import numpy as np
import glob
import torch


def data_classes(name="camvid"):
    camvid_classes = ['Background', 'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
    camvid_cmap = np.array([[0, 0, 0],
                    [128, 128, 128],
                    [128, 0, 0],
                    [192, 192, 192],
                    [128, 64, 128],
                    [60, 40, 222],
                    [128, 128, 0],
                    [192, 128, 128],
                    [64, 64, 128],
                    [64, 0, 128],
                    [64, 64, 0],
                    [0, 128, 192],
                    ])
    cityscapes_classes = ['unlabeled', 'road','sidewalk', 'building','wall', 'fence','pole', 'traffic_light','traffic_sign',
                          'vegetation','terrain','sky', 'person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle',]
    cityscapes_cmap = np.array([
                    [0, 0, 0],
                    [128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]
    ])

    if name == "camvid":
        return camvid_classes, camvid_cmap
    elif name == 'cityscapes':
        return cityscapes_classes, cityscapes_cmap
    elif name == 'aeroscapes':
        return cityscapes_classes, cityscapes_cmap
    else:
        raise NameError("No such dataset!")


def calculate_class_weights(Y, n_classes, method="paszke", c=1.02):
    """ Given the training data labels Calculates the class weights.

    Args:
        Y:      (numpy array) The training labels as class id integers.
                The shape does not matter, as long as each element represents
                a class id (ie, NOT one-hot-vectors).
        n_classes: (int) Number of possible classes.
        method: (str) The type of class weighting to use.

                - "paszke" = use the method from from Paszke et al 2016
                            `1/ln(c + class_probability)`
                - "eigen"  = use the method from Eigen & Fergus 2014.
                             `median_freq/class_freq`
                             where `class_freq` is based only on images that
                             actually contain that class.
                - "eigen2" = Similar to `eigen`, except that class_freq is
                             based on the frequency of the class in the
                             entire dataset, not just images where it occurs.
                -"logeigen2" = takes the log of "eigen2" method, so that
                            incredibly rare classes do not completely overpower
                            other values.
        c:      (float) Coefficient to use, when using paszke method.

    Returns:
        weights:    (numpy array) Array of shape [n_classes] assigning a
                    weight value to each class.

    References:
        Eigen & Fergus 2014: https://arxiv.org/abs/1411.4734
        Paszke et al 2016: https://arxiv.org/abs/1606.02147
    """
    # CLASS PROBABILITIES - based on empirical observation of data
    ids, counts = np.unique(Y, return_counts=True)
    n_pixels = Y.size
    p_class = np.zeros(n_classes)
    p_class[ids] = counts/n_pixels

    # CLASS WEIGHTS
    if method == "paszke":
        weights = 1/np.log(c+p_class)
    elif method == "eigen":
        assert False, "TODO: Implement eigen method"
        # TODO: Implement eigen method
        # where class_freq is the number of pixels of class c divided by
        # the total number of pixels in images where c is actually present,
        # and median freq is the median of these frequencies.
    elif method in {"eigen2", "logeigen2"}:
        epsilon = 1e-8 # to prevent division by 0
        median = np.median(p_class)
        weights = median/(p_class+epsilon)
        if method == "logeigen2":
            weights = np.log(weights+1)
    else:
        assert False, "Incorrect choice for method"

    return weights


def decode_labels(mask, cmap=None):
    cmap = cmap
    h, w = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(mask)
    for i in range(1):
        for val in unique:
            outputs[mask == val] = cmap[val]
    return outputs

def cal_mean_std(img_paths):
    mean = np.array([0, 0, 0])
    std = np.array([0, 0, 0])
    for path in img_paths:
        img = imread(path)
        for i in range(3):
            mean[i] += img[:, :, i].mean()
            std[i] += img[:, :, i].std()

    mean = mean / len(img_paths)
    std = std / len(img_paths)
    return mean, std


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save
    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".
    """
    name = args.name
    save_dir = args.save_dir

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path+'.pth')

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save
    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.
    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.
    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


if __name__ == '__main__':
    paths = glob.glob(r'E:\datasets\CamVid480x360\trainannot\*.png')
    paths.extend(glob.glob(r'E:\datasets\CamVid480x360\testannot\*.png'))
    paths.extend(glob.glob(r'E:\datasets\CamVid480x360\valannot\*.png'))
    # label = []
    # for path in paths:
    #     lbl = imread(path)
    #     label.append(lbl)
    # label = np.array(label)
    # print(label.shape)
    # weights = calculate_class_weights(label, 12)
    # print(weights)
#     paths = glob.glob(r'E:\datasets\CamVid480x360\train\*.png')
#     paths.extend(glob.glob(r'E:\datasets\CamVid480x360\test\*.png'))
#     paths.extend(glob.glob(r'E:\datasets\CamVid480x360\val\*.png'))
#     mean, std = cal_mean_std(paths)
#     print(mean, std)