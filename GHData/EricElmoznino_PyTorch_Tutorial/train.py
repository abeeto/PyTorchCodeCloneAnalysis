import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
from tensorboardX import SummaryWriter

from evaluate import evaluate_dataset, classification_accuracy
from models import Model, PretrainedModel
from MNISTDataset import MNISTDataset
import utils

# Training configuration
use_custom_model = True
n_epochs = 2
batch_size = 16
lr = 1e-3

# Create the model
if use_custom_model:
    model = Model(in_channels=3, n_classes=10)
    model_name = 'custom_model'
else:
    model = PretrainedModel(n_classes=10)
    model_name = 'pretrained_model'

# One of the few downsides of PyTorch is it doesn't automatically detect if a gpu is available
if torch.cuda.is_available():
    model.cuda()

# Create the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Instantiate the dataloaders
train_set = MNISTDataset('data/train', resolution=[32, 32],training=True)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
test_set = MNISTDataset('data/test', resolution=[32, 32])
test_loader = DataLoader(test_set, num_workers=2)

# Prepare save directories and configure tensorboard logging
shutil.rmtree(os.path.join('saved_runs', model_name), ignore_errors=True)
os.mkdir(os.path.join('saved_runs', model_name))
writer = SummaryWriter(os.path.join('saved_runs', model_name, 'logs'))

print_freq = 50
running_losses = None
for epoch in range(1, n_epochs + 1):
    print('Starting epoch %d\n' % epoch)

    for batch, data in enumerate(train_loader):
        step_num = utils.step_num(epoch, batch, train_loader)   # Global step

        # Obtain the inputs/targets
        images, labels = data
        labels = labels.view(-1)    # Labels must be 1 dimensional with class indices, but DataLoader creates batch dimension
        images, labels = utils.device([images, labels])     # Send to gpu if possible

        # Compute the losses
        predictions = model(images)
        loss = F.cross_entropy(predictions, labels)

        # Backprop and optimize
        optimizer.zero_grad()   # Zero out the .grad properties on all model parameters
        loss.backward()         # Backpropagate the error and populate the .grad properties
        optimizer.step()        # Optimizer uses the .grad values to modify the parameters

        # Compute error metrics and update running losses
        accuracy = classification_accuracy(predictions, labels)
        losses = {'loss': loss.item(), 'accuracy': accuracy}    # .item() obtains the raw value of a 0D Tensor
        running_losses = utils.update_losses(running_losses, losses, print_freq)

        if (step_num + 1) % print_freq == 0:
            utils.log_to_tensorboard(writer, running_losses, step_num)
            utils.print_losses(running_losses, step_num, n_epochs * len(train_loader))
            running_losses = {l: 0 for l in running_losses}

    print('Finished epoch %d\n' % epoch)

    print('Evaluation')
    eval_losses = evaluate_dataset(model, test_loader)
    utils.log_to_tensorboard(writer, eval_losses, step_num, training=False)
    utils.print_losses(eval_losses, step_num, n_epochs * len(train_loader))

    utils.save_model(model, os.path.join('saved_runs', model_name))
