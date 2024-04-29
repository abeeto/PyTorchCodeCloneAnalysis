import torch
import matplotlib.pyplot as plt
import numpy as np
import config
from data_loader import *
from model import *
from utils import *
from loss import *

# Multislice Viewer by: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data

def multi_slice_viewer(img, gt, pred):
    fig, ax = plt.subplots(1, 2)
    gt_c = img_tumor_map(img, gt)
    pred_c = img_tumor_map(img, pred)
    for i in range(2):
        if i == 1:
            ax[i].volume = pred_c
        else:
            ax[i].volume = gt_c
        ax[i].index = img.shape[0] // 2
        if i == 1:
            ax[i].imshow(pred_c[ax[i].index])
        else:
            ax[i].imshow(gt_c[ax[i].index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def img_tumor_map(img, map):
    color_0 = [0/255, 0/255, 0/255, 0]
    color_1 = [150/255, 10/255, 50/255, 0]
    color_2 = [10/255, 150/255, 50/255, 0]
    color_3 = [10/255, 100/255, 100/255, 0]
    map_0 = map == 0
    map_1 = map == 1
    map_2 = map == 2
    map_3 = map == 3
    map_ = np.zeros([128, 128, 128, 4])
    map_[map_0, :] = np.array(color_0)
    map_[map_1, :] = np.array(color_1)
    map_[map_2, :] = np.array(color_2)
    map_[map_3, :] = np.array(color_3)
    print(map_)
    img = img - np.min(img)
    img = img / np.max(img)
    map_[..., 3] = img
    return map_

def process_key(event):
    fig = event.canvas.figure
    ax_0 = fig.axes[0]
    ax_1 = fig.axes[1]
    if event.key == 'j':
        previous_slice(ax_0)
        previous_slice(ax_1)
    elif event.key == 'k':
        next_slice(ax_0)
        next_slice(ax_1)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def cod2label(map):
    map_com = torch.zeros([128, 128, 128])
    wt_mask = map[0] == 1
    map_com[wt_mask] = 1
    et_mask = map[1] == 1
    map_com[et_mask] = 2
    ed_mask = map[2] == 1
    map_com[ed_mask] = 3
    return map_com


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.animation as animation

    csv_path = '/media/tensorist/Extreme SSD/brats2020/testset.csv'
    loader = BrainLoader(csv_path, train=False)
    model = UNet3D(4, [16, 32, 64, 128, 256], num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    train, test = dataset_loader()
    load_checkpoint(config.CHECKPOINT + '_72.pth.tar', model, optimizer, optimizer.param_groups[0]['lr'])
    loss_metric = LossAndMetric(num_classes=4)

    animation = False

    frames = []

    for i in range(1):
        img, gt = loader[1]
        img = torch.tensor(img[np.newaxis, ...])
        gt = torch.tensor(gt[np.newaxis, ...])
        img = img.to(config.DEVICE)
        gt = gt.to(config.DEVICE)
        pred_logits = model(img.float())
        pred_probs = torch.sigmoid(pred_logits)
        pred_probs_one_mask = pred_probs > 0.5
        pred_ = torch.zeros_like(pred_probs)
        pred_[pred_probs_one_mask] = 1
        pred_ = pred_.squeeze()
        img = img.cpu().numpy().squeeze()
        gt = gt.cpu().numpy().squeeze()
        gt_com = cod2label(gt)
        pred_com = cod2label(pred_)
        if animation:
            fig, ax = plt.subplots(1, 2)
            gt_c = img_tumor_map(img[0], gt_com)
            pred_c = img_tumor_map(img[0], pred_com)
            for i in range(img.shape[1]):
                frames.append([ax[0].imshow(gt_c[i, :, :], animated=True),
                               ax[1].imshow(pred_c[i, :, :], animated=True)])
        else:
            multi_slice_viewer(img[0], gt_com, pred_com)
            plt.show()

    if animation:
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                        repeat_delay=1000)
        ani.save('/home/tensorist/project_files/tumor_segmentation/application_movie_bsp8.gif', writer='imagemagick')
        plt.show()

