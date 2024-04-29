#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import argparse

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")      # Apple Silicon Metal
except:
    DEVICE = torch.device("cpu")

BATCHSIZE = 100
DIR = os.path.join(os.environ["HOME"], ".pytorch")
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 100
N_VALID_EXAMPLES = BATCHSIZE * 30

class CNN(nn.Module):

    def __init__(self, kernel_size, n_cnn1, n_cnn2, n_ln, dropout1, dropout2):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_cnn1, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(n_cnn1, n_cnn2, kernel_size=kernel_size, stride=1, padding=0)

        if kernel_size == 3:
            n_ln0 = int(n_cnn2 * 12 * 12)

        if kernel_size == 5:
            n_ln0 = int(n_cnn2 * 10 * 10)
            
        self.fc1 = nn.Linear(n_ln0, n_ln)
        self.fc2 = nn.Linear(n_ln, 10)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        
    def forward(self, x):
        batch_size = x.shape[0]

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = self.dropout1(h)

        h = torch.flatten(h, 1)

        h = F.relu(self.fc1(h))
        h = self.dropout2(h)
        h = self.fc2(h)
        
        return F.log_softmax(h, dim=1)


def define_model(trial):
    # カーネルサイズ
    kernel_size = trial.suggest_int('kernel_size', 3, 5, 2)

    # CNN数
    n_cnn1 = trial.suggest_int('n_cnn1', 10, 200)
    n_cnn2 = trial.suggest_int('n_cnn2', 10, 200)

    # ドロップアウト
    dropout1 = trial.suggest_float("dropout1", 0.2, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.2, 0.5)

    # MLP数
    n_ln = trial.suggest_int('n_ln', 100, 2000)
    
    return CNN(kernel_size, n_cnn1, n_cnn2, n_ln, dropout1, dropout2)


def get_mnist():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break

                data, target = data.to(DEVICE), target.to(DEVICE)
                
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Optuna, MNIST")
    parser.add_argument('--trial', type=int, default=10, metavar='N',
                        help='number of trial (default: 10)')
    args = parser.parse_args()
    
    n_trials = args.trial
    study_name ="mnist-cnn"

    study = optuna.create_study(pruner=optuna.pruners.PercentilePruner(50),
                                storage="sqlite:///result/optuna.db",
                                study_name=study_name,
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()


# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
