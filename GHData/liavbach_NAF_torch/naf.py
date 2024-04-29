import os

import torch
import torch.nn as nn
from torch.optim import Adam

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(device_name)
DTYPE = torch.double


def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def update_target_model(target, source, tau=1.0):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QNetwork(nn.Module):

    def __init__(self, hidden_size, state_features_size, action_space):
        super(QNetwork, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # batch network, layer 0 , size num_inputs = state features
        self.bn0 = nn.BatchNorm1d(state_features_size).to(device=DEVICE)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(state_features_size, hidden_size).to(device=DEVICE)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(device=DEVICE)

        # linear function, receives hidden_size and output the estimated value function
        self.V = nn.Linear(hidden_size, 1).to(device=DEVICE)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # linear function, returns the best action for a state
        self.mu = nn.Linear(hidden_size, num_outputs).to(device=DEVICE)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        # in order to calculate the Advantage function
        self.L = nn.Linear(hidden_size, num_outputs ** 2).to(device=DEVICE)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        # lower triangular matrix (without the diagonal), for calculate the advantage function
        self.tril_mask = torch.tril(
            torch.ones(num_outputs, num_outputs, dtype=DTYPE, device=DEVICE),
            diagonal=-1).unsqueeze(0)
        # for advantage function calculation
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs, dtype=DTYPE, device=DEVICE))).unsqueeze(0)

    def forward(self, inputs):
        x, u = inputs  # state, action
        x = self.bn0(x)
        x = torch.relu_(self.linear1(x))
        x = torch.relu_(self.linear2(x))

        V = self.V(x)  # applying linear1, linear2 on x and finally V.
        mu = torch.tanh(self.mu(x))  # applying linear1, linear2 on x and finally F.than(mu(x)).

        Q = None
        # calculating the advantage function
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)

            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)

            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF:
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = QNetwork(hidden_size, num_inputs, action_space).to(DEVICE, dtype=DTYPE)
        self.target_model = QNetwork(hidden_size, num_inputs, action_space).to(DEVICE, dtype=DTYPE)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        update_target_model(self.target_model, self.model)

    # returns action normalized to range of [-1,1]
    def select_action(self, state, action_noise=None):
        state = torch.tensor([state], dtype=DTYPE, device=DEVICE)
        self.model.eval()
        mu, _, _ = self.model((state, None))
        self.model.train()
        mu = mu.data
        if action_noise is not None:
            mu += torch.tensor(action_noise.noise(), dtype=DTYPE, device=DEVICE)

        return mu.clamp(-1, 1).cpu().numpy()[0]

    def update_parameters(self, batch):
        batch_size = len(batch.state)
        state_batch = torch.tensor(batch.state, dtype=DTYPE, device=DEVICE).view(batch_size, -1)
        action_batch = torch.tensor(batch.action, dtype=DTYPE, device=DEVICE).view(batch_size, -1)
        reward_batch = torch.tensor(batch.reward, dtype=DTYPE, device=DEVICE).view(batch_size, -1)
        mask_batch = torch.tensor(batch.mask, dtype=DTYPE, device=DEVICE).view(batch_size, -1)
        next_state_batch = torch.tensor(batch.next_state, dtype=DTYPE, device=DEVICE).view(batch_size, -1)

        _, _, next_state_values = self.target_model((next_state_batch, None))  # V' (of theta - target model)

        # expected_state_action_values = reward_batch + (self.gamma * mask_batch + next_state_values) bug?
        expected_state_action_values = reward_batch + (self.gamma * mask_batch * next_state_values)

        _, state_action_values, _ = self.model((state_batch, action_batch))

        loss = mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        update_target_model(self.target_model, self.model, self.tau)

        return loss.item()

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/naf_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(model_path))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            print('Loading model from {}'.format(model_path))
            self.model.load_state_dict(torch.load(model_path))
            self.target_model.load_state_dict(torch.load(model_path))

