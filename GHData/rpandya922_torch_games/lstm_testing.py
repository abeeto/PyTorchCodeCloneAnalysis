# TODO: delete this file once testing is done, it should become an option in lta_features to use or not
from collections import OrderedDict

import gym
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from BoSEnv import RepeatedBoSEnv
from utils import helpful_partner, adversarial_partner, bach_partner, stravinsky_partner
from autoencoder import create_dataset2
from utils import train

class HumanLSTM(nn.Module):
    def __init__(self, state_size, hidden_size, human_out, sequence_length, num_layers=2):
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.human_out = human_out
        self.sequence_length = sequence_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.state_size, 
            hidden_size=self.hidden_size, 
            # num_layers=self.num_layers,
            # batch_first=True
        )
        # self.linear1 = nn.Linear(self.sequence_length*4, self.hidden_size*self.sequence_length)
        # self.linear = nn.Linear(self.hidden_size*self.sequence_length, self.human_out)
        # self.linear = nn.Linear(self.hidden_size*self.num_layers, self.human_out)

        self.linear = nn.Linear(self.hidden_size, self.human_out)

    def forward(self, x):
        x1, (hn, _) = self.lstm(x)

        # x1 = self.linear1(x.flatten(start_dim=1))
        # x2 = self.linear(th.unsqueeze(x1.flatten(), 0))
        x2 = self.linear(hn.squeeze(0))
        return F.log_softmax(x2, dim=1)
        # return self.linear(hn.swapaxes(0, 1).flatten(start_dim=1))


class HumanFC(nn.Module):
    def __init__(self, state_size, hidden_size, human_out, sequence_length):
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.human_out = human_out
        self.sequence_length = sequence_length

        self.linear1 = nn.Linear(self.state_size*self.sequence_length, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.human_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.linear1(x.flatten()))
        x2 = self.linear2(x1)
        return F.log_softmax(x2, dim=1)

class SequenceDataset(Dataset):
    def __init__(self, data, labels, sequence_length=5):
        self.X = th.tensor(data).float()
        self.y = th.tensor(labels).float()
        self.sequence_length = sequence_length

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = th.cat((padding, x), 0)

        return x, self.y[i]


def create_labels(states, actions):
    inputs = states[:-1,:,:]
    labels = actions[1:,:,:]
    
    return inputs, labels


def zero_pad_episodes(data, padding=5):
    """Adds 0-padding between episodes so LSTM doesn't consider data between episodes as contiguous
    """
    data_padded = []
    for i in range(data.shape[2]):
        episode = data[:,:,i]
        data_padded.append(np.vstack((episode, np.zeros((padding, episode.shape[1])))))
    return np.vstack(data_padded)

if __name__ == "__main__":
    partner_policies = [helpful_partner]
    state_size = 4
    hidden_size = 32
    human_out = 2
    sequence_length = 5
    horizon = 2
    num_layers = 1

    # model = HumanLSTM(
    #     state_size=state_size, 
    #     hidden_size=hidden_size, 
    #     human_out=human_out,
    #     sequence_length=sequence_length,
    #     num_layers=num_layers
    # )

    model = HumanFC(state_size=state_size, 
        hidden_size=hidden_size, 
        human_out=human_out,
        sequence_length=sequence_length,
    )

    # training set creation
    states, actions = create_dataset2(partner_policies, n_datapoints=20, onehot_actions=False)
    inputs, labels = create_labels(states, actions)
    inputs = zero_pad_episodes(inputs, padding=sequence_length-1)
    labels = zero_pad_episodes(labels, padding=sequence_length-1)
    dataset = SequenceDataset(inputs, labels)
    trainset_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # validation set creation
    states, actions = create_dataset2(partner_policies, n_datapoints=20, onehot_actions=False)
    inputs, labels = create_labels(states, actions)
    inputs = zero_pad_episodes(inputs, padding=sequence_length-1)
    labels = zero_pad_episodes(labels, padding=sequence_length-1)
    dataset = SequenceDataset(inputs, labels)
    valset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # training parameters
    epoch = 300
    lr = 0.1
    optimizer = th.optim.SGD(model.parameters(), lr=lr)

    all_train_loss, all_val_loss = train(model, optimizer, trainset_loader, valset_loader, epoch=epoch, loss_fn=nn.NLLLoss)

    print(all_train_loss)
    import ipdb; ipdb.set_trace()
