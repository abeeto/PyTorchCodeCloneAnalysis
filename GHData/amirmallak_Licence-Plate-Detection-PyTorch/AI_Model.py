import pandas as pd

from plot_results import plot
from saving_model import save_model
from loading_model import load_model
from train_neural_network import train
from neural_network_model import neural_network
from creating_training_val_test_datasets import _create_datasets


def build_AI_model(dataset: pd.DataFrame, channel: int, width: int, height: int) -> None:
    dataset_params = _create_datasets(dataset, width, height)
    train_loader, val_loader, test_loader, test_batch_loader, train_dataset, val_dataset, test_dataset = dataset_params

    model, train_step, loss_fn = neural_network(channel, width, height)

    train_results = train(model, train_loader, train_dataset, val_loader, val_dataset, train_step, loss_fn, width)
    training_accuracies, training_losses, validation_accuracies, validation_losses = train_results

    save_model(model)

    # Loading previous ready-trained models
    model = load_model()

    plot(model, width, height, train_loader, train_dataset, training_accuracies, training_losses, validation_accuracies,
         validation_losses, test_dataset, test_loader, test_batch_loader)
