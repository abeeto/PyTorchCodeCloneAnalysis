import torch
import torch.nn as nn
import csv
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from math import exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_graph(out_path, epoch, train_records, val_records, test_sets_records, test_sets_names, label):
    epochs = range(1, epoch+2)
    plt.plot(epochs, train_records, 'g', label='Training '+label)
    plt.plot(epochs, val_records,   'b', label='Validation '+label)
    colors = ['y', 'k', 'r', 'c', 'm', 'w']
    colors = colors[0:len(test_sets_names)]
    for test_records, set_name, color in zip(test_sets_records, test_sets_names, colors):
        plt.plot(epochs, test_records,  color, label=set_name+"-"+label)
    plt.title('Training Validation and Test '+label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(out_path)
    # plt.show()
    plt.close()
