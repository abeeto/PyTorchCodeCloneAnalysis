import numpy as np
import matplotlib.pyplot as plt

import pickle

g_losses = np.load("g_losses.npy")
d_losses = np.load("d_losses.npy")

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(d_losses, label='Discriminator')
ax.plot(g_losses, label='Generator')

plt.title("Training")
plt.legend()
plt.show()