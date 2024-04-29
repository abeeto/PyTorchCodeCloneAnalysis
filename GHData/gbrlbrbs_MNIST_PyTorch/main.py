import torch
import os
from utils import get_data, get_whole_model, training_loop, evaluate_pictures, plot_confusion_matrix
from constants import *


def run():
    train_loader, valid_loader, train_dataset, valid_dataset = get_data()
    if os.path.exists(SAVE_PATH):
        evaluate_pictures(valid_dataset)
        plot_confusion_matrix(valid_loader)
    else:
        net, criterion, optimizer = get_whole_model()
        net, _, _ = training_loop(net, criterion, optimizer, train_loader, valid_loader)
        evaluate_pictures(valid_dataset, net)
        plot_confusion_matrix(valid_loader, net)


if __name__ == '__main__':
    run()
