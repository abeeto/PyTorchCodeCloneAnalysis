import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
torch.set_num_threads(1)

class PolicyFeedforward(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.scaler = StandardScaler()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

    def do_train(self, x, y, args):
        batch_size = args.batch_size
        num_batch = len(x) // batch_size + (1 if len(x) % batch_size else 0)

        self.scaler.fit(x)
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_func = nn.MSELoss()

        for i in range(args.epoch):
            self.train()
            epoch_loss = 0
            for j in tqdm(list(range(num_batch))):
                x_batch, y_batch = x[j*batch_size:(j+1)*batch_size], y[j*batch_size:(j+1)*batch_size]
                x_batch = self.scaler.transform(x_batch)
                x_batch, y_batch = torch.FloatTensor(x_batch).to(device), torch.FloatTensor(y_batch).to(device).unsqueeze(1)
                y_out = self(x_batch)

                loss = loss_func(y_out, y_batch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(f'epoch {i}, loss {epoch_loss/num_batch}, score {self.score(x, y, args)}')
            print(f'epoch {i}, loss {epoch_loss/num_batch}')
            if args.model_dir:
                os.makedirs(args.model_dir, exist_ok=True)
                self.save(os.path.join(args.model_dir, f'model_{i}.pt'))
                self.save(os.path.join(args.model_dir, f'model.pt'))

    def score(self, x, y, args):
        batch_size = args.batch_size
        num_batch = len(x) // batch_size + (1 if len(x) % batch_size else 0)

        y_tmp = torch.FloatTensor(y)
        u, v = 0, ((y_tmp - y_tmp.mean()) ** 2).sum().item()
        self.eval()
        with torch.no_grad():
            for i in range(num_batch):
                x_batch, y_batch = x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
                x_batch = self.scaler.transform(x_batch)
                x_batch, y_batch = torch.FloatTensor(x_batch).to(device), torch.FloatTensor(y_batch).to(device).unsqueeze(1)
                y_out = self(x_batch)
                u += (y_batch - y_out).pow(2).sum().item()

        return (1 - u/v)

    @staticmethod
    def load(path):
        if use_cuda == 'cuda':
            model_file = torch.load(path)
        else:
            model_file = torch.load(path, map_location='cpu')
        model = PolicyFeedforward(model_file['input_dim'], model_file['hidden_dim'])
        model.load_state_dict(model_file['state_dict'])
        model.scaler = model_file['scaler']

        return model

    def save(self, path):
        torch.save({
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'state_dict': self.state_dict(),
            'scaler': self.scaler,
        }, path)