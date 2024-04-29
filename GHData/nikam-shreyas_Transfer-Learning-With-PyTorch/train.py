import argparse
import sys

import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import KMNIST

from model import LeNetModel

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_SPLIT = 0.80
VAL_SPLIT = 1 - TRAIN_SPLIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    """
    Function to parse the command-line arguments
    :return: the paths to the output trained model and the corresponding plots
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True,
                    help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())


def load_data():
    """
    Function to load the KMNIST dataset and generate the training, validation and test datasets.
    :return: the training, validation and test dataloaders respectively
    """
    # Loading the dataset from the torchvision datasets
    train_data = KMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = KMNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())

    # Split the training dataset based on the training/validation split
    num_train_samples = int(len(train_data) * TRAIN_SPLIT)
    num_val_samples = int(len(train_data) * VAL_SPLIT)
    train_data, validation_data = random_split(train_data, [num_train_samples, num_val_samples],
                                               torch.Generator().manual_seed(42))

    # Create the data loaders for the datasets
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_data_loader, validation_data_loader, test_data_loader


def train():
    args = get_args()
    train_data_loader, validation_data_loader, test_data_loader = load_data()

    # Calculate the training and validation steps per epoch
    train_steps = len(train_data_loader.dataset) // BATCH_SIZE
    validation_steps = len(validation_data_loader.dataset) // BATCH_SIZE

    # Loading the model
    print('[INFO] Loading the LeNet Model')
    model = LeNetModel(input_channels=1, output_classes=len(train_data_loader.dataset.classes)).to(device)

    # Setting the loss function as the Negative Log Likelihood Loss and the optimiser as the Adam optimiser
    criterion = torch.nn.NLLLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create a dictionary to store the training stats
    training_history = {
        'training_losses': [],
        'validation_losses': [],
        'training_accuracy': [],
        'validation_accuracy': []
    }

    for epoch in range(EPOCHS):

        # Set the model in training mode
        model.train()

        # Initialize the variables to store the training stats
        total_train_loss = 0
        total_correct = 0
        total_validation_loss = 0
        total_validation_correct = 0
        total_test_loss = 0
        total_test_correct = 0

        # Iterate through the images and their labels
        for images, labels in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            output_ps = model(images)
            loss = criterion(output_ps, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_train_loss += loss
            total_correct += (output_ps.argmax(1) == labels).type(torch.float).sum().item()

        model.eval()
        with torch.no_grad():
            for images, labels in validation_data_loader:
                images, labels, images.to(device), labels.to(device)
                output_ps = model(images)
                loss = criterion(output_ps, labels)
                total_validation_loss += loss
                total_validation_correct += (output_ps.argmax(1) == labels).type(torch.float).sum().item()

        # Calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_validation_loss = total_validation_loss / validation_steps

        # Calculate the training and validation accuracy
        total_correct = total_correct / len(train_data_loader.dataset)
        total_validation_correct = total_validation_correct / len(validation_data_loader.dataset)

        # Update our training history
        training_history["training_loss"].append(avg_train_loss.cpu().detach().numpy())
        training_history["training_accuracy"].append(total_correct)
        training_history["validation_loss"].append(avg_validation_loss.cpu().detach().numpy())
        training_history["validation_accuracy"].append(total_validation_correct)

        # Print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, total_correct))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avg_validation_loss, total_validation_correct))

        test_predictions = []
        with torch.no_grad():
            for images, labels in test_data_loader:
                images, labels, images.to(device), labels.to(device)
                output_ps = model(images)
                loss = criterion(output_ps, labels)
                total_test_loss += loss
                total_test_correct += (output_ps.argmax(1) == labels).type(torch.float).sum().item()

                test_predictions.extend(output_ps.argmax(axis=1).cpu().numpy())

        print(classification_report(test_data_loader.dataset.targets.targets.cpu().numpy(),
                                    np.array(test_predictions), target_names=test_data_loader.dataset.classes))

        # Plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(training_history["train_loss"], label="train_loss")
        plt.plot(training_history["val_loss"], label="val_loss")
        plt.plot(training_history["train_acc"], label="train_acc")
        plt.plot(training_history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])

        # Serialize the model to disk
        torch.save(model, args["model"])
