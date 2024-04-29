'''The saved TIFF format using quandequan cannot be opened with most
image viewer, so we'd better use this tool to open it.'''
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import os
from quandequan import functional as QF

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('file')
    config = parser.parse_args()
    return config

def main(config):
    img = QF.read_img(config.file)
    plt.imshow(img.transpose((1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    config = parse_config()
    main(config)

