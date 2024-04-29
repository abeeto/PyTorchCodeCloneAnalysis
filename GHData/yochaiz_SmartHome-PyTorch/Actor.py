from DeepNetwork import DeepNetwork
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
import torch


class Actor(DeepNetwork):
    def __init__(self, stateDim, actionDim, policy, gpus):
        self.gpus = gpus

        self.stateDim = stateDim
        self.actionDim = actionDim

        self.layersSize = [self.stateDim] + ([128] * 5) + [self.actionDim]

        super(Actor, self).__init__()

        # exploration params
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01  # exploration minimal rate
        self.epsilon_decay = 0.99

        # init policy obj
        self.policy = policy
        # number of knn neighbors to compare when converting continuous action to discrete action
        self.k = self.policy.nActions
        # init knn object
        self.knn = NearestNeighbors(n_neighbors=self.k)
        # init knn object train set
        self.knn.fit(self.policy.possibleActions)

    def initModelLayers(self):
        layers = []
        for i in range(len(self.layersSize) - 2):
            layers.append(nn.Linear(self.layersSize[i], self.layersSize[i + 1]))
            layers.append(nn.ReLU())

        i += 1
        layers.append(nn.Linear(self.layersSize[i], self.layersSize[i + 1]))
        layers.append(nn.Tanh())

        return layers

    def toSequential(self):
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model.forward(x)

    # stateVar is a variable
    def __optimalActionPerState(self, stateVar, criticModel, validActions, discreteAction, i):
        # duplicate state as number of actions for Q-value prediction
        stateVar.data = stateVar.data.expand(validActions.size(0), self.stateDim)
        # calc Q-value for each valid action
        validActionsVariable = Variable(validActions).cuda(self.gpus)
        # TODO: currently state is being normalized while action isn't
        Qvalues = criticModel.forward([stateVar, validActionsVariable]).data
        # choose highest Q-value action
        actionID = np.argmax(Qvalues)
        # select optimal valid action
        discreteAction[i] = validActions[actionID]

    # state - original state, i.e. unnormalized & as tensor
    def wolpertingerAction(self, state, criticModel):
        # normalize state
        stateNorm = self.policy.normalizeState(state)
        # create variable for state
        stateNormVar = Variable(stateNorm).cuda(self.gpus)
        # predict continuous action
        contAction = self.forward(stateNormVar).data
        # find IDs of closest (discrete, possible) actions ranked from closest to farthest
        rankedActions = self.knn.kneighbors(contAction, return_distance=False)
        # evaluate Qvalues for each state
        nSamples = state.size(0)
        # init optimal discrete action for each state
        discreteAction = torch.zeros(nSamples, self.actionDim).type_as(state)
        # select optimal discrete action for each state
        for i in range(nSamples):
            # convert IDs to the actions themselves
            validActions = self.policy.possibleActions[rankedActions[i].tolist()]
            # filter valid actions per state
            validActions = self.policy.filterValidActionsPerState(state[i], validActions)
            # select optimal action per state
            self.__optimalActionPerState(stateNormVar[i], criticModel, validActions, discreteAction, i)

        return discreteAction

    # state is a tensor
    def act(self, state, criticModel):
        isRandom = int(np.random.rand() <= self.epsilon)

        if isRandom > 0:
            while True:
                action = self.policy.generateRandomAction()
                if self.policy.isActionValid(state, action):
                    break
        else:
            # unsqueeze change shape to (1,n) where n is the state dimension (length)
            state = state.unsqueeze(0)
            # find optimal discrete action
            action = self.wolpertingerAction(state, criticModel)
            action = action[0]

        return action, bool(isRandom)

    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, value):
        assert (0 <= value <= 1)
        self.epsilon = value

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, (self.epsilon * self.epsilon_decay))
