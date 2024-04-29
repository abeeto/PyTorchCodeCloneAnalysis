import torch
import numpy as np
import matplotlib.pyplot as plt

def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy

def plot_result(y, constant = None):
    x = torch.arange(0, len(y)).numpy()
    y = np.array(y)
    plt.figure()
    if constant is not  None:
        plt.plot(x, 0 * x + constant)
    plt.plot(x, y)
    plt.show()
