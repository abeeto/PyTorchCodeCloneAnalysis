# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:17:25 2020

@author: IVAN
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def graphics_show_loss_acc(saved_file, min_epoch=1, max_epoch=20,
                           min_loss=0, max_loss=20, min_accuracy=0.0, max_accuracy=1.0):
    # читаем из файла
    losses = list()
    accuracies = list()

    with open(saved_file) as file:
        losses_accuracies_list = file.read()

    losses_accuracies_list = losses_accuracies_list.split('\n')
    losses_accuracies_list = [value for value in losses_accuracies_list if value]

    print(losses_accuracies_list)
    rows_count = len(losses_accuracies_list)
    for i in range(2, rows_count):
        row = losses_accuracies_list[i].split(':')
        accuracies.append(float(row[0]))
        losses.append(float(row[1]))

    print(losses)
    print(accuracies)
    epochs_count = len(losses)
    epochs = [x + 1 for x in range(epochs_count)]
    # for i in range(0,epochs_count):
    #     print('''Epoch {}:Average Loss ={:.3f}
    #           Average Accuracy = {:.3f}'''
    #           .format( epochs[i],losses[i],accuraces[i]))        

    fig, axes = plt.subplots(2, 1)
    # график значения функции потерь на каждой эпохе
    axes[0].plot(epochs, losses)
    axes[0].grid(which='major',
                 color='k')

    axes[0].grid(which='minor',
                 color='gray',
                 linestyle=':')
    axes[0].set_ylim(min_loss, max_loss)
    axes[0].set_xlim(min_epoch, max_epoch)

    axes[0].set_xlabel("Эпоха")  # ось абсцисс
    axes[0].set_ylabel("Функция потерь")  # ось ординат
    axes[0].set_title("Значение функции потерь при обучении")

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
    axes[1].set_ylim(min_accuracy, max_accuracy)
    axes[1].set_xlim(min_epoch, max_epoch)

    axes[1].set_xlabel("Эпоха")  # ось абсцисс
    axes[1].set_ylabel("Точность распознавания")  # ось ординат
    axes[1].set_title("Значение средней точности распознавания при обучении")

    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[1].minorticks_on()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig(saved_file + ".png")


print("Input Report Filename:")
report_file = str(input()).strip('"')
graphics_show_loss_acc(report_file, max_epoch=20, max_loss=1, min_accuracy=0.7)
