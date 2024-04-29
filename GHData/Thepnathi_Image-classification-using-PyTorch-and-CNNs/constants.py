"""
Assignment 2, COMP338 - Step 0 - This file contains the constants and general helper functions

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import itertools
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from imgdata import imageDataset, DefaultTrainSet, DefaultTestSet
from cnn import ConvolutionalNetwork

class Constants(object):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    default_batch_size = 16
    num_epochs = [5, 20]
    learning_rates = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06]

    train_dataset = imageDataset('data', 'img_list_train.npy')
    test_dataset = imageDataset('data', 'img_list_test.npy')

    n_training_samples = len(train_dataset)
    n_test_samples = len(test_dataset)

    # Random indices
    train_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    test_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    CLASSES = ['faces', 'dog', 'airplanes', 'keyboard', 'cars']

    TRAIN_HISTORY_FNAME = 'train_history_dict.npy'

    line = "=================="

# The following function plot_confusion_matrix is taken from Tutorial/Labs 7 & 8 by Prof. Shan Luo
class Plot_Tools(object):
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        :param cm: (numpy matrix) confusion matrix
        :param classes: [str]
        :param normalize: (bool)
        :param title: (str)
        :param cmap: (matplotlib color map)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def load_trained_models():
    # Stores all the trained cnn models
    trained_models = {}

    # Iterate through learning rates and stores the trained model by learning rate
    for num_epochs in Constants.num_epochs:
        for rate in Constants.learning_rates:
            net = ConvolutionalNetwork()
            net.load_state_dict(th.load(gen_model_fname(rate, num_epochs)))
            trained_models[rate] = net

    return trained_models

def gen_model_fname(learning_rate, num_epochs):
    return f"trained_models/model_epochs-{num_epochs}_learning_rate-" + "{:.0e}".format(learning_rate) + ".pth"

def display_predictions(class_type, correct, incorrect, title=""):
    rows = 2
    cols = max(len(correct), len(incorrect)) + 2
    f, axarr = plt.subplots(rows, cols, num=None, figsize=(10, 6))
    f.canvas.set_window_title(title)
    mpl.rcParams['toolbar'] = 'None'

    for r in range(rows):
        axarr[r,0].axis('off')
        axarr[r,0].set_title("Correct" if r == 0 else "Incorrect")

        for c in range(1, cols):
            axarr[r,c].axis('off')
            # Correct predictions in first and incorrect in second row.
            if r == 0 and c < len(correct):
                axarr[r,c].imshow(correct[c]['im'])
            elif r == 1 and c < len(incorrect):
                axarr[r,c].imshow(incorrect[c]['im'])


    # SHow and cloase after a button is pressed.
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

