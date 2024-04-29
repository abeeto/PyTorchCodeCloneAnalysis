import os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import sub


def cdf(array):
    num_bins = len(array)
    counts, bin_edges = np.histogram(array, bins=num_bins, normed=True)
    cdf = np.cumsum(counts) / sum(counts)
    return bin_edges[:-1], cdf


if __name__ == '__main__':

    cdf_arrays = []
    labels = ['usingTorchSubprocess', 'usingPyTorch']

    for csv_file in ['results/old/measurements.csv', 'results/new/measurements.csv']:
        with open(csv_file) as f:
            lines = f.readlines()

        latencies = []
        for line in lines:
            latencies.append(float(line.rstrip()))
        
        cdf_arrays.append(latencies)

    plt.figure(0)
    for idx, array in enumerate(cdf_arrays):
        x, y = cdf(array)
        plt.plot(x, y, label=labels[idx])

    plt.xlim(0)
    plt.legend()
    plt.savefig('results/cdf.png')
