import os

import torch.nn as nn
import torch.optim as optim

from data_handling.base import BaseDataLoader
from data_handling.example_CIFAR10Dataset import ExampleCIFAR10Dataset
from models.example_simple_net import ExampleSimpleNet
from trainer.base_trainer import BaseTrainer
from trainer.example_CIFAR10Predictor import ExampleCIFAR10Predictor

"""
This file contains an very simple example main file to show how
to the framework to build, train, validate, save, load a model
and save predictions.

Note that the base trainer integrates basic tensorboard support
(training loss and validation loss are recorded).
"""

# Define the Torch dataset
cifar10_dataset = ExampleCIFAR10Dataset(
    images_path="C:\\Users\\t-mebern\\Downloads\\cifar-10\\train\\train",
    has_labels=True,
    label_path="C:\\Users\\t-mebern\\Downloads\\cifar-10\\trainLabels.csv")

# Create the corresponding dataloader for training and validation
data_loader = BaseDataLoader(dataset=cifar10_dataset,
                             batch_size=16,
                             validation_split=0.1)

# Define the model
model = ExampleSimpleNet()
save_dir = os.getcwd()+'\\runs\\exp1'

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize the trainer
trainer = BaseTrainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      data_loader=data_loader,
                      n_epochs=100,
                      save_dir=save_dir,
                      save_model_frequency=1)

# Start training
trainer.train()

# Test model loading
trained_model = ExampleSimpleNet()
trained_model.restore(save_dir, 'model-1.pt')

# Predict on the test set and save the predictions
cifar10_testset = ExampleCIFAR10Dataset(
    images_path="C:\\Users\\t-mebern\\Downloads\\cifar-10\\test\\test",
    has_labels=False)
predictor = ExampleCIFAR10Predictor(trained_model, cifar10_testset)
outputs = predictor.predict(save_path=save_dir)
