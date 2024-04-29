from DeepNetwork import DeepNetwork
import torch.nn as nn
import torch


class Critic(DeepNetwork):
    def __init__(self, stateDim, actionDim):
        self.stateDim = stateDim
        self.actionDim = actionDim

        self.statePathHidden = [self.stateDim, 256, 128]
        self.mergedPathHidden = [self.statePathHidden[-1]] + ([128] * 5) + [1]

        super(Critic, self).__init__()

        self.state_model = None
        self.action_model = None
        self.merged_model = None

    def initModelLayers(self):
        # init state separate layers
        stateLayers = []
        i = 0
        for i in range(len(self.statePathHidden) - 2):
            stateLayers.append(nn.Linear(self.statePathHidden[i], self.statePathHidden[i + 1]))
            stateLayers.append(nn.ReLU())

        if len(stateLayers) > 0:
            i += 1
        stateLayers.append(nn.Linear(self.statePathHidden[i], self.statePathHidden[i + 1]))

        # init action separate layers
        actionLayers = [nn.Linear(self.actionDim, self.statePathHidden[-1])]

        # init merged path layers
        mergedLayers = []
        i = 0
        for i in range(len(self.mergedPathHidden) - 2):
            mergedLayers.append(nn.Linear(self.mergedPathHidden[i], self.mergedPathHidden[i + 1]))
            mergedLayers.append(nn.ReLU())

        if len(mergedLayers) > 0:
            i += 1
        mergedLayers.append(nn.Linear(self.mergedPathHidden[i], self.mergedPathHidden[i + 1]))

        return stateLayers, actionLayers, mergedLayers

    def toSequential(self):
        stateLayers, actionLayers, mergedLayers = self.layers

        self.state_model = nn.Sequential(*stateLayers)
        self.action_model = nn.Sequential(*actionLayers)
        self.merged_model = nn.Sequential(*mergedLayers)

    def forward(self, x):
        state, action = x

        out1 = self.state_model.forward(state)
        out2 = self.action_model.forward(action)
        v = torch.add(out1, out2)

        return self.merged_model.forward(v)
