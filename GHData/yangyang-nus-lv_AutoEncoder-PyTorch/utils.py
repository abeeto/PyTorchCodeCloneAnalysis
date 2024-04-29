import numpy as np
import matplotlib.pyplot as plt


def to_img(x):
    """transform a tensor to image."""
    x = x.view(x.size(0), 1, 28, 28)
    return x
