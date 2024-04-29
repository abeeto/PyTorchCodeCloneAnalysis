import numpy as np
from PIL import Image
import os


def read_data(folder):
    files = os.listdir(folder)
    x = []
    y = []
    for file in files:
        img = Image.open(f"{folder}/{file}")
        r, g, b = img.split()
        x.append([np.array(r) / 255, np.array(g) / 255])
        y.append(int(file[-5]))

    return np.array(x), np.array(y)
