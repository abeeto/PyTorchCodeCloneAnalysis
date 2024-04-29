import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from LossLogger import LossLogger

class LossPlotter:
    def __init__(self, xlabel, ylabel, title):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def plot(self, train_losses, test_losses, window=1):
        fig, ax = plt.subplots()
        x_len = len(train_losses)

        x = np.arange(1, x_len + 1)
        y = train_losses
        ax.plot(x, y, label='Train')

        x = np.arange(1, x_len + 1)
        y = test_losses
        ax.plot(x, y, label='Test')

        ax.set(
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            title=self.title
        )
        ax.grid()
        # horizontal line at y=0
        ax.axhline(y=0, color='k')
        #fig.savefig("test.png")
        ax.legend()
        plt.show()

# import losses from file
logger = LossLogger()
train_losses, test_losses = logger.read_losses('losses/losses.csv')

plotter = LossPlotter('Epoch', 'Loss', 'Language Categorization')
plotter.plot(train_losses, test_losses)
