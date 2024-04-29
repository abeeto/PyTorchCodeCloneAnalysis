import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt



def gen_impuls():
    steps = np.linspace(0, 1, 10, dtype=np.float32)  # float32 for converting torch FloatTensor
    x_np = np.power(steps, 2)
    steps = np.linspace(1, 0, 10, dtype=np.float32)
    x_np_1 = np.power(steps, 2)
    return np.concatenate((x_np, x_np_1))


def gen_zero():
    return np.zeros(20, dtype=np.float32)



def gen_hart(n):
    res = gen_zero()
    for i in range(n):
        res = np.concatenate((res, gen_impuls()))
        res = np.concatenate((res, gen_zero()))

    return res


data = gen_hart(10)
plt.plot(data)
plt.show()