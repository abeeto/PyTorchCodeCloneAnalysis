

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        train = read('./result/train_loss.txt')
        test = read('./result/test_loss.txt')
        plt.plot(train, 'g', label='train')
        plt.plot(test, 'r', label='validation')
        plt.legend(loc='upper right')


    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')
        plt.plot(bleu, 'r', label='bleu score')
        plt.legend(loc='upper right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
