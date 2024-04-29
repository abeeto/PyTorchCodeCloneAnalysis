import click
import torch
import time
import pandas as pd
import random
from tqdm import tqdm

# Custom defined
# from models import *
from utils import data_utils, test_utils


@click.group()
def cli():
    pass


def _train(model, device, lr,
           train_dataloader, val_dataloader, test_dataloader,
           epochs, model_save_path):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_curve_info = []
    for epoch in range(epochs):
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward Propagation
            y_pred = model.predict(batch_x)

            # Compute and print loss
            loss = criterion(y_pred, batch_y.float())

            # Zero the gradients
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

        acc = test_utils.batch_accuracy(model, val_dataloader)
        is_better = len(loss_curve_info) == 0 or \
            acc > max(
                loss_curve_info,
                key=lambda x: x['val_acc'])['val_acc']

        if (is_better):
            print("Saving model...")
            model.save(model_save_path + "/model.bin")

            torch.save(
                optimizer.state_dict(),
                model_save_path + "/model.bin.optim"
            )

        loss_curve_info.append(
            {"epoch": epoch, "loss": loss.item(), "val_acc": acc}
        )
        print('epoch: ', epoch, ' loss: ', loss.item(), 'acc: ', acc)

    acc = test_utils.batch_accuracy(model, test_dataloader)
    print('Test accuracy: ', acc)

    loss_curve_info = pd.DataFrame(loss_curve_info)
    loss_curve_info.to_csv(
        "charts/" + str(int(time.time())) + ".csv",
        index=False
    )


@cli.command()
@click.option('--batch_size', default=10)
@click.option('--hidden', default=50)
@click.option('--lr', default=0.001)
@click.option('--save_to', default="saved_models/", help="dir for models")
@click.option('--epochs', default=10, help="number of epochs")
def train(batch_size, hidden, lr, save_to, use_small, epochs):
    """Run the training loop.

    Usage: python main.py train [flags]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    ###############################################
    # LOAD DATA
    ###############################################
    print("=" * 50)
    datasets, dataloaders = \
        data_utils.get_train_test_dataloaders(batch_size)
    train_dataset, val_dataset, test_dataset = datasets
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    ###############################################
    # LOAD MODEL
    ###############################################
    print("=" * 50)
    model = None  # import or construct model
    model = model.to(device)

    ###############################################
    # START TRAINING
    ###############################################
    print("=" * 50)
    _train(model, device, lr,
           train_dataloader, val_dataloader, test_dataloader,
           epochs, save_to)


@cli.command()
@click.option('--saved_to', default="saved_models/", help="dir for models")
def test(hidden, use_model, lr, saved_to):
    """Run the model on test data if applicable.

    Usage: python main.py test [flags]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    cli()
