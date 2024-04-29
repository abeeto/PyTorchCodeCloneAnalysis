# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:17:25 2020

@author: IVAN
"""
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def graphics_show_loss_acc(losses, accuracies, save_file):
    epochs_count = len(losses)
    epochs = [x + 1 for x in range(epochs_count)]
    for i in range(0, epochs_count):
        print('''Epoch {}:Average Loss ={:.3f}
              Average Accuracy = {:.3f}'''
              .format(epochs[i], losses[i], accuracies[i]))

    fig, axes = plt.subplots(2, 1)
    # график значения функции потерь на каждой эпохе

    max_y_val = int(ceil(max(losses))) + 1
    max_x_val = len(losses) + 1

    axes[0].plot(epochs, losses)
    axes[0].grid(which='major',
                 color='k')

    axes[0].grid(which='minor',
                 color='gray',
                 linestyle=':')
    axes[0].set_ylim(0, max_y_val)
    axes[0].set_xlim(0, max_x_val)

    axes[0].set_xlabel("Эпоха")  # ось абсцисс
    axes[0].set_ylabel("Потеря")  # ось ординат
    axes[0].set_title("График функции потерь")

    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axes[0].minorticks_on()

    # график точности на каждой эпохе

    axes[1].plot(epochs, accuracies)
    axes[1].grid(which='major',
                 color='k')

    axes[1].grid(which='minor',
                 color='gray',
                 linestyle=':')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_xlim(0, max_x_val)

    axes[1].set_xlabel("Эпоха")  # ось абсцисс
    axes[1].set_ylabel("Точность")  # ось ординат
    axes[1].set_title("График точности")

    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[1].minorticks_on()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig(save_file)
