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


DEVICE = torch.device("cpu")
BATCHSIZE = 100
DIR = os.path.join(os.environ["HOME"], ".pytorch")
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 100
N_VALID_EXAMPLES = BATCHSIZE * 30

class LSTM(nn.Module):

    def __init__(self, hidden_dim, lstm_layers, dropout):
        
        super(LSTM, self).__init__()
        self.seq_len = 28               # 縦方向を時系列のSequenceとしてLSTMに入力する
        self.feature_dim = 28           # 横方向ベクトルをLSTMに入力する
        self.hidden_dim = hidden_dim    # 隠れ層のサイズ
        self.lstm_layers = lstm_layers  # LSTMのレイヤー数　(LSTMを何層重ねるか)

        self.lstm = nn.LSTM(self.feature_dim, 
                            self.hidden_dim, 
                            num_layers = self.lstm_layers,
                            dropout = dropout)
        
        self.fc = nn.Linear(self.hidden_dim, 10) # MNIST: 10クラス分類

    # h, c の初期値
    def init_hidden_cell(self, batch_size):
        hedden = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim)
        cell = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim)        
        return (hedden, cell)

    def forward(self, x):
        batch_size = x.shape[0]

        hidden_cell = self.init_hidden_cell(batch_size)
        h1 = x.view(batch_size, self.seq_len, self.feature_dim) # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
        h2 = h1.permute(1, 0, 2)                                # (Batch, Seqence, Feature) -> (Seqence , Batch, Feature)

        lstm_out, (h_n, c_n) = self.lstm(h2, hidden_cell) # LSTMの入力データのShapeは(Seqence, Batch, Feature)
                                                                 
        x = h_n[-1,:,:] # tn-1ののlstm_layersの最後のレイヤーを取り出す  (B, h)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


def define_model(trial):
    # 隠れ層のユニット数
    hidden_dim = trial.suggest_int("hidden_dim", 30, 150)

    # LSTM層数
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)

    # ドロップアウト
    if lstm_layers == 1:
        dropout = 0
    else:
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
    
    return LSTM(hidden_dim, lstm_layers, dropout)


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

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

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
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
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
    study_name ="mnist-lstm"

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
