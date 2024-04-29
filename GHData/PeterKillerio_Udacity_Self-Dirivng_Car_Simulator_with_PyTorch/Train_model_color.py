# Import libraries
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import files
import model_architecture_color
import prepare_data_color

# Search for cuda card and activate if available
print("Cuda is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the cpu")

# Create model and define hyperparameters
################################################################################
MODEL_NAME = f"model-{int(time.time())}"
model = (model_architecture_color.CNN()).to(device)
model.train()

lr = 0.0009
epochs = 26
batch_size = 240

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Learning rate reduction
#
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.90)

# Return learning rate: code from https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

################################################################################

# Create train and test loaders loaded from prepare_data.py file which we imported
train_loader = DataLoader(dataset = prepare_data_color.dataset_train, batch_size = batch_size)
test_loader = DataLoader(dataset = prepare_data_color.dataset_test, batch_size = batch_size)

# Define training function
def train(model,train_loader, test_loader, optimizer, criterion, epochs, batch_size):
    for epoch in range(epochs):
        for x, y, z in train_loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            optimizer.zero_grad()
            yhat = (model(x, z))
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

        loss_valid = 0.0
        for x, y, z in test_loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            yhat = model(x, z)
            loss = criterion(yhat, y)

            # If loss is zero one possibility is that test_loader is empty so you need
            # to check the number of images you are reding in your 'prepare_dataset.py' file
            loss_valid += loss

        # Decrease learning rate
        # scheduler.step()
        # print("Decreasing learning rate to", get_lr(optimizer))


        print(f"epoch: {epoch+1}, validation loss: {loss_valid}")

# Start training
train(model, train_loader, test_loader, optimizer, criterion, epochs, batch_size)
# Save the parameters
torch.save(model.state_dict(), f"{MODEL_NAME}_color.pt")
