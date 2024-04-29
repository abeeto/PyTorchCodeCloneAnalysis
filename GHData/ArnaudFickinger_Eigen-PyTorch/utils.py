import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from options import Options
import matplotlib.pyplot as plt
import numpy as np

opt = Options().parse()

class ResizeImgAndDepth(object):
    def __init__(self, size_tup):
        self.size = size_tup

    def __call__(self, sample):
        img = Image.fromarray(sample['image'], 'RGB').resize(self.size)
        depth = Image.fromarray(sample['depth'], 'L').resize((self.size[0] // 2, self.size[1] // 2))
        return {'image': np.array(img), 'depth': np.array(depth)}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        depth = sample["depth"]
        if np.random.random() > 0.5:
            img = np.fliplr(sample['image']).copy()
            depth = np.fliplr(sample['depth']).copy()
        return {'image': img, 'depth': depth}


class ImgAndDepthToTensor(object):
    def __init__(self):
        self.ToTensor = transforms.ToTensor()

    def __call__(self, sample):
        return {'image': self.ToTensor(sample['image']), 'depth': torch.tensor(sample['depth'], dtype=torch.float)}


class NormalizeImg(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, sample):
        return {'image': self.normalize(sample['image']), 'depth': sample['depth']}


class UnNormalizeImgBatch(object):
    def __init__(self, mean, std):
        self.mean = mean.reshape((1, 3, 1, 1))
        self.std = std.reshape((1, 3, 1, 1))

    def __call__(self, batch):
        return (batch * self.std) + self.mean

def print_training_loss_summary(loss, loss_array, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    # prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps % n_batches)

    if (steps_this_epoch == 1 or steps_this_epoch % print_every == 0):
        print ('Epoch [{}/{}], Iteration [{}/{}], Total Loss: {:.4f}'
               .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss)+str(loss_array))


def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_image_tensor_in_subplot(ax, img_tensor):
    im = img_tensor.cpu().numpy().transpose((1, 2, 0))
    # pil_im = Image.fromarray(im, 'RGB')
    ax.imshow(im)


def plot_depth_tensor_in_subplot(ax, depth_tensor):
    im = depth_tensor.cpu().numpy()
    # im = im*255
    # im = im.astype(np.uint8)
    # pil_im = Image.fromarray(im, 'L')
    ax.imshow(im, 'gray')

def plot_result(images, depths, preds, figname, rmse, logrmse, eigen):
    fig, axes = plt.subplots(4, 3)

    for i in range(3):
        plot_image_tensor_in_subplot(axes[i, 0], images[i])
        plot_depth_tensor_in_subplot(axes[i, 1], depths[ i])
        plot_depth_tensor_in_subplot(axes[i, 2], preds[ i])
        hide_subplot_axes(axes[i, 0])
        hide_subplot_axes(axes[i, 1])
        hide_subplot_axes(axes[i, 2])
    axes[4, 0].plot(np.arange(len(rmse)), rmse)
    axes[4, 0].set_title('RMSE')
    axes[4, 1].plot(np.arange(len(logrmse)), logrmse)
    axes[4, 1].set_title('logRMSE')
    axes[4, 2].plot(np.arange(len(eigen)), eigen)
    axes[4, 2].set_title('Eigen')
    plt.savefig(figname)

def plot_model_predictions_on_sample_batch(images, depths, preds, figname, plot_from=0, figsize=(12, 12)):

    fig, axes = plt.subplots(5, 3, figsize=figsize)

    for i in range(4):
        plot_image_tensor_in_subplot(axes[i, 0], images[plot_from + i])
        plot_depth_tensor_in_subplot(axes[i, 1], depths[plot_from + i])
        plot_depth_tensor_in_subplot(axes[i, 2], preds[plot_from + i])
        hide_subplot_axes(axes[i, 0])
        hide_subplot_axes(axes[i, 1])
        hide_subplot_axes(axes[i, 2])
    plt.savefig(figname)