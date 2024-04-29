"""
Assignment 2, COMP338 - Step 3. Train the Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import time
import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from constants import Constants, gen_model_fname
from cnn import ConvolutionalNetwork, createLossAndOptimizer

def plot_training_history(train_history):
    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    ax1.set_title('Evoloution of the training loss.')
    ax2.set_title('Evoloution of the training accuracy.')

    for rate, (loss_hist, accuracy_hist) in train_history.items():
        label = "{:.0e}".format(rate) + " learning rate"
        x = np.arange(1, len(loss_hist) + 1)
        ax1.plot(x, loss_hist, label=label)
        ax2.plot(x, accuracy_hist, label=label)

    ax1.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
    ax2.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))

    # Single legend for both plots.
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    plt.setp(ax1, xlabel='Epoch', ylabel='Loss')
    plt.setp(ax2, xlabel='Epoch', ylabel='Accuracy')
    plt.show()

def train(net, batch_size, n_epochs, learning_rate):
    """
    Train a neural network, print statistics of the training and save the trained model to
    the file 'model_learning_rate_{learning_rate}.pth'

    Return the training history.

    :param net: (PyTorch Neural Network)
    :param batch_size: (int)
    :param n_epochs: (int)  Number of iterations on the training set
    :param learning_rate: (float) learning rate used by the optimizer
    """
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    criterion, optimizer = createLossAndOptimizer(net, learning_rate)

    # Init variables used for plotting the loss and accuracy
    train_history = []
    accuracy_history = []

    training_start_time = time.time()

    n_minibatches = len(Constants.train_dataset) // batch_size
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        start_time = time.time()

        total_train_loss = 0
        total_accurate = 0

        for i in range(n_minibatches):

            # Gather data for this mini batch
            inputs = th.tensor([Constants.train_dataset[j]['imNorm'] for j in range(i, i+batch_size)], dtype=th.float32)
            labels = th.tensor([Constants.train_dataset[j]['label'] for j in range(i, i+batch_size)], dtype=th.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # outputs has raw scores for each class, argmax is used to get the index of the highest
            # score, i.e. the predicted label.
            total_accurate += th.sum(th.argmax(outputs, dim=1) == labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss_in_epoch = total_train_loss / len(Constants.train_dataset)
        accuracy_in_epoch = total_accurate / len(Constants.train_dataset)
        train_history.append(average_loss_in_epoch)
        accuracy_history.append(accuracy_in_epoch)

        model_fname = gen_model_fname(learning_rate, epoch+1)
        th.save(net.state_dict(), model_fname)

        # Print a single line of statistinc after every epoch.
        print(f'Epoch: {epoch + 1}', end='\t')
        print(f'average loss: {average_loss_in_epoch}', end='\t')
        print(f'training accuracy: {accuracy_in_epoch}', end='\t')
        print(f'took: {time.time() - start_time}s')

    print("Training Finished, took {:.2f}s".format(time.time() - training_start_time))

    # Load the trained model into the network
    net.load_state_dict(th.load(model_fname))

    return train_history, accuracy_history


if  __name__ == "__main__":
    # Don' train if we have existing models.
    if not os.path.isfile(Constants.TRAIN_HISTORY_FNAME):
        # Each learning rate gets its own training history.
        # The training history consists of the loss and accuracy values for each epoch of training.
        train_history = {}
        for rate in Constants.learning_rates:
            train_history[rate] = train(ConvolutionalNetwork(), batch_size=16,
                                        n_epochs=max(Constants.num_epochs), learning_rate=rate)

        np.save(Constants.TRAIN_HISTORY_FNAME, train_history)

    # Plot the loss and accuracy for different learning rates.
    train_history = np.load(Constants.TRAIN_HISTORY_FNAME, allow_pickle=True)[()]
    plot_training_history(train_history)
