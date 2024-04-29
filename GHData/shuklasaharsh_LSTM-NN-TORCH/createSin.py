import numpy as np
import matplotlib.pyplot as plt

N = 100  # Number of Samples
L = 1000  # Length of Samples
T = 20


def getSineWave(numberOfSamples, lengthOfSamples, T):
    x = np.empty((numberOfSamples, lengthOfSamples), np.float32)  # Empty numpy array
    x[:] = np.array((range(lengthOfSamples))) + np.random.randint(-4 * T, 4 * T, numberOfSamples).reshape(numberOfSamples, 1)
    y = np.sin(x / 1.0 / T).astype(np.float32)
    plt.figure(figsize=(10, 8))
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(np.arange(x.shape[1]), y[0, :], 'r', linewidth=2.0)
    # plt.show()

    return {"x": x, "y": y, "plt": plt}
