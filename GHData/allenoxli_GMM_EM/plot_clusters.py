import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

all_plots = []

class PlotAllClusters:
    def __init__(self, epochs):
        self.all_plots = []
        self.fig = plt.figure(figsize=(45,45))
        self.row = math.ceil(epochs/10)
        self.col = math.ceil(epochs/self.row)

    def plot_clusters(self, x, k_components, mu, var, plot_num):
        x = x.squeeze(1)
        mu = mu.squeeze(0)
        var = var.squeeze(0)
        colors = ['#0000C6', '#FF5809', '#E1E100', '#9F4D95', '#EA0000', '#00FFFF']

        self.fig.add_subplot(self.row, self.col, plot_num+1)
        plt.axis([-10, 10, -5, 10])
        plt.scatter(x[:,0], x[:,-1], s=1, c='#408080')
        ax = plt.gca()

        for i in range(k_components):
            ellipse_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': '-'}
            ellipse = Ellipse(mu[i], 3 * var[i][0], 3 * var[i][1], **ellipse_args)
            ax.add_patch(ellipse)

    def show_plot(self):
        plt.savefig('processing.png')
        plt.show()
