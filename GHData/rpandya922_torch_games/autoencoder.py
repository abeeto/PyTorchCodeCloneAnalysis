import gym
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from BoSEnv import RepeatedBoSEnv
from utils import helpful_partner, adversarial_partner, train

class Autoencoder(nn.Module):
    def __init__(self, n_input, n_latent, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_latent = n_latent
        self.n_output = n_output

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, n_latent),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 16),
            nn.ReLU(),
            nn.Linear(16, n_output)
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))

class AEDataset(Dataset):
    def __init__(self, states, actions):

        x = np.hstack((states, actions))
        y = np.array(actions)

        self.data = th.from_numpy(x).float()
        self.labels = th.from_numpy(y).float()
        self.len = len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x, self.labels[index]

    def __len__(self):
        return self.len

def dummy_robot_policy(obs, env):
    return env.action_space.sample()

def create_dataset2(partner_policies, horizon=20, n_datapoints=10000, onehot_actions=True):
    # TODO: combine with function below
    env = RepeatedBoSEnv(partner_policies, horizon)
    obs = env.reset()

    robot_policy = lambda obs: dummy_robot_policy(obs, env)

    all_states = [[]]
    all_h_actions = [[]]

    for i in range(n_datapoints):
        robot_action = robot_policy(obs)
        next_obs, rew, done, _ = env.step(robot_action)

        all_states[-1].append(obs)
        all_h_actions[-1].append(next_obs[1])

        obs = next_obs
        if done:
            obs = env.reset()
            all_states.append([])
            all_h_actions.append([])

    # one-hot encode states 
    states = np.array(all_states[:-1])
    states2 = np.dstack((np.eye(2)[states[:,:,0]], np.eye(2)[states[:,:,1]]))
    states2 = states2.swapaxes(0, 1).swapaxes(1, 2)

    states = states.swapaxes(0, 1).swapaxes(1, 2)
    states_onehot = []
    for j in range(states.shape[2]):
        # only care about first two elements of state
        states_onehot.append(np.hstack((np.eye(2)[states[:,0,j]], np.eye(2)[states[:,1,j]])))
    states = np.dstack(states_onehot)

    # one-hot encode actions
    actions = np.array(all_h_actions[:-1])
    actions2 = np.eye(2)[np.array(all_h_actions[:-1])]
    actions2 = actions2.swapaxes(0, 1).swapaxes(1, 2)

    actions = actions.swapaxes(0, 1)
    actions_onehot = []
    for j in range(actions.shape[1]):
        # only care about first two elements of state
        actions_onehot.append(np.eye(2)[actions[:,j]])

    if onehot_actions:
        actions = np.dstack(actions_onehot)
    else:
        actions = np.expand_dims(actions, 1)

    # print((states == states2).all(), (actions == actions2).all())

    return states, actions

    # return states.swapaxes(0, 1).swapaxes(1, 2), actions.swapaxes(0, 1).swapaxes(1, 2)

def create_dataset(partner_policies, horizon=20, n_datapoints=10000):

    env = RepeatedBoSEnv(partner_policies, horizon)
    obs = env.reset()

    robot_policy = lambda obs: dummy_robot_policy(obs, env)

    all_states = []
    all_h_actions = []

    for i in range(n_datapoints):
        robot_action = robot_policy(obs)
        next_obs, rew, done, _ = env.step(robot_action)

        all_states.append(obs)
        all_h_actions.append(next_obs[1])

        obs = next_obs
        if done:
            obs = env.reset()

    # one hot encode states (since it just consists of actions)
    states = np.array(all_states)
    states = np.hstack((np.eye(2)[states[:,0]], np.eye(2)[states[:,1]]))

    # one hot encode actions
    actions = np.eye(2)[np.array(all_h_actions)]

    return states, actions

def train_and_save(partner_policies):
    states, actions = create_dataset(partner_policies, n_datapoints=8000)
    trainset = AEDataset(states, actions)
    trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    states, actions = create_dataset(partner_policies, n_datapoints=2000)
    valset = AEDataset(states, actions)
    valset_loader = DataLoader(valset, batch_size=32, shuffle=True)

    ## TRAINING PARAMS
    epoch = 70
    lr = 5e-3

    # input is states and actions, output is just actions
    model = Autoencoder(n_input=6, n_latent=1, n_output=2)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    all_train_loss, all_val_loss = train(model, optimizer, trainset_loader, valset_loader, epoch)

    # save model
    th.save(model.state_dict(), "./data/autoencoder.pt")


if __name__ == "__main__":
    partner_policies = [helpful_partner, adversarial_partner]

    # train_and_save(partner_policies)

    states, actions = create_dataset(partner_policies, n_datapoints=8000)
    trainset = AEDataset(states, actions)
    trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    states, actions = create_dataset(partner_policies, n_datapoints=2000)
    valset = AEDataset(states, actions)
    valset_loader = DataLoader(valset, batch_size=32, shuffle=True)

    ## TRAINING PARAMS
    epoch = 70
    lr = 5e-3

    # input is states and actions, output is just actions
    model = Autoencoder(n_input=6, n_latent=1, n_output=2)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    all_train_loss, all_val_loss = train(model, optimizer, trainset_loader, valset_loader, epoch)


    env = RepeatedBoSEnv([helpful_partner], horizon=20)
    obs = env.reset()

    robot_policy = lambda obs: dummy_robot_policy(obs, env)

    all_states = []
    all_h_actions = []

    for i in range(10):
        robot_action = robot_policy(obs)
        next_obs, rew, done, _ = env.step(robot_action)

        if i > 0:
            import ipdb; ipdb.set_trace()

        obs = next_obs