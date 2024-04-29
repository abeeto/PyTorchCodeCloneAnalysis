import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.pyplot.switch_backend("Agg")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")              # M1 Metal
    else:
        return torch.device("cpu")  # No Accelerator


def save_results(history) -> None:
    accuracy = [res["acc"] for res in history]
    losses = [res["loss"] for res in history]
    val_accuracy = [res["val_acc"] for res in history]
    val_losses = [res["val_loss"] for res in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(losses, "-o", label="Loss")
    ax1.plot(val_losses, "-o", label="Validation Loss")
    ax1.legend()

    ax2.plot(100 * np.array(accuracy), "-o", label="Accuracy")
    ax2.plot(100 * np.array(val_accuracy), "-o", label="Validation Accuracy")
    ax2.legend()
    plt.savefig("./static/results.png")


if __name__ == "__main__":
    print(get_device())
