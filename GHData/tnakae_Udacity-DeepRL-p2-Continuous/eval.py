import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def plot(fpath):
    with open(fpath, "rb") as fp:
        scores = pickle.load(fp)

    episodes = np.arange(len(scores)) + 1
    scores = pd.Series(scores)
    scores_ma = scores.rolling(100).mean()

    plt.figure(figsize=[8,4])
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(episodes, scores, c="blue")
    plt.plot(episodes, scores_ma, c="red", linewidth=3)
    plt.axhline(30, linestyle="--", color="black", linewidth=2)
    plt.grid(which="major")
    plt.legend(["Episode Average Score", "Moving Average Score (100 Episodes)",
                "Criteria"])
    plt.tight_layout()
    plt.savefig("./bestmodel_score.png")

if __name__ == "__main__":
    plot("./rewards.pickle")
