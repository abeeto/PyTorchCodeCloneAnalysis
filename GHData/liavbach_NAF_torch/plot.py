import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_results():
    results = os.listdir('results')
    num_results = len(results)

    fig, ax = plt.subplots(nrows=num_results, ncols=1, figsize=(6, 4 * num_results))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, result in enumerate(results):
        env_name = result[8:-4]

        df = pd.read_csv(os.path.join('results', result))
        ax[i].plot(df['Episode'], df['Score'])
        ax[i].set_title(env_name)

    plt.show()


if __name__ == '__main__':
    plot_results()
