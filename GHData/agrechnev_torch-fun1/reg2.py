# By Oleksiy Grechnyev, IT-JIM on 6/25/20.
# Here I try the boston dataset (load via sklearn)

import sys
import torch

from sklearn.datasets import load_boston

########################################################################################################################

class BostonSet(torch.utils.data.Dataset):
    def __init__(self, mode):
        # Load boston
        self.boston = load_boston()

        x0, y0 = self.boston.data, self.boston.target

        if mode == 'train':
            x, y = x0[:450], y0[:450]
        elif mode == 'val':
            x, y = x0[450:], y0[450:]
        else:
            raise ValueError('BostonSet : Wrong mode !')
        # Normalize, use full x0, y0 !
        self.x = torch.tensor((x - x0.mean(axis=0)) / x0.std(axis=0)).float()
        self.y = torch.tensor((y - y0.mean()) / y0.std()).float().reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return (self.x[item], self.y[item])

########################################################################################################################


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Probably needs some regularization !
        self.fc1 = torch.nn.Linear(13, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

########################################################################################################################

class Trainer:
    def __init__(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.n_epoch = 200

        # train set + loader
        self.train_set = BostonSet('train')
        print(f'len(train_set) = {len(self.train_set)}')
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # val set + loader
        self.batch_size = 5
        self.val_set = BostonSet('val')
        print(f'len(val_set) = {len(self.train_set)}')
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Network
        self.net = Net().to(self.device)
        print(self.net)

        # Criterion + optimize
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self):
        self.net.train()
        for epoch in range(self.n_epoch):
            loss_sum = 0.
            for (i, (batch_x, batch_y)) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(batch_x)
                loss = self.criterion(out, batch_y)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
            loss_train = loss_sum / i
            loss_val = self.validate()
            print(f'{epoch} : loss_train = {loss_train}, loss_val = {loss_val}')

    def validate(self):
        self.net.eval()
        loss_sum = 0.

        with torch.no_grad():
            for (i, (batch_x, batch_y)) in enumerate(self.val_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                out = self.net(batch_x)
                loss = self.criterion(out, batch_y)
                loss_sum += loss.item()
            loss_val = loss_sum / i

        return loss_val

    def check_for_const(self):
        self.net.eval()
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.numpy()
            with torch.no_grad():
                pred = self.net(x).cpu().numpy()
            print('pred.shape = ', pred.shape)
            print('pred = ', pred)
            print('y = ', y)


########################################################################################################################


trainer = Trainer()
trainer.train()
trainer.check_for_const()