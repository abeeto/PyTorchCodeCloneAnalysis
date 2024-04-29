''' This deals with continious image
'''

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import os
from quandequan import functional as QF

NUM_BIN = 513

ALLOWED_FORMAT = ['.png', '.bmp', '.jpg', 'spi', '.tiff']

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, help='the base image dir')
    parser.add_argument('-r', '--screen', action='store_true',
            help='if set, the plot will be shown on screen')
    parser.add_argument('-a', '--save', type=str, default='',
            help='if set, the plot will save to the file')
    config = parser.parse_args()
    return config

def main(config):
    base_path = Path(config.dir)
    acc0 = np.array([0 for i in range(NUM_BIN)])   # RGB
    acc1 = np.array([0 for i in range(NUM_BIN)])   # RGB
    acc2 = np.array([0 for i in range(NUM_BIN)])   # RGB
    edge = None
    for root, dirs, files in os.walk(base_path, topdown=True):
        for name in files:
            filename = os.path.join(root, name)
            if not Path(filename).suffix in ALLOWED_FORMAT:
                continue
            npimg = QF.read_img(filename)
            if npimg.dtype != np.float32 and npimg.dtype != np.float64:
                npimg = np.array(npimg, dtype=np.float32) / 255
            # print(npimg.shape)
            imghist, edge = np.histogram(npimg[0, :, :], bins=NUM_BIN, range=(0, 1))
            acc0 += imghist
            imghist, edge = np.histogram(npimg[1, :, :], bins=NUM_BIN, range=(0, 1))
            acc1 += imghist
            imghist, edge = np.histogram(npimg[2, :, :], bins=NUM_BIN, range=(0, 1))
            acc2 += imghist

    plt.figure(num=None, figsize=(50, 6), dpi=100, facecolor='w', edgecolor='k')
    eps = 1 / NUM_BIN / 3
    x = edge[:NUM_BIN]
    plt.hist(x + eps, bins=edge, weights=acc0, color='r', histtype='step')
    plt.hist(x + eps, bins=edge, weights=acc1, color='g', histtype='step')
    plt.hist(x + eps, bins=edge, weights=acc2, color='b', histtype='step')
    if config.screen:
        plt.show()
    if config.save:
        plt.savefig(config.save)




if __name__ == "__main__":
    config = parse_config()
    main(config)

