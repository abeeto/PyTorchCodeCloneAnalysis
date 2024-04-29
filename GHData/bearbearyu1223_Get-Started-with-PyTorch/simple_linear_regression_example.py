import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split


class CustomDataset(Dataset):
    def __init__(self, _x_tensor, _y_tensor):
        assert isinstance(_x_tensor, torch.FloatTensor)
        assert isinstance(_y_tensor, torch.FloatTensor)
        self.x = _x_tensor
        self.y = _y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class LinearRegression(nn.Module):
    def __init__(self, feature_dim, target_dim):
        super().__init__()
        assert isinstance(feature_dim, int)
        assert isinstance(target_dim, int)
        self.linear = nn.Linear(in_features=feature_dim, out_features=target_dim)

    def forward(self, _x):
        assert isinstance(_x, torch.FloatTensor)
        return self.linear(_x)


def make_train_step(_model, _loss_fn, _optimizer):
    def train_step_fn(_x, _y):
        _model.train()
        _y_hat = _model(_x)
        _loss = _loss_fn(_y, _y_hat)
        _loss.backward()
        _optimizer.step()
        _optimizer.zero_grad()
        return _loss.item()

    return train_step_fn


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(42)
    x = np.random.rand(1000, 1)
    true_a, true_b = 1, 2
    y = true_a + true_b * x + 0.1 * np.random.randn(1000, 1)
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = CustomDataset(_x_tensor=x_tensor, _y_tensor=y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)

    model = LinearRegression(1, 1).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    train_step = make_train_step(_model=model, _loss_fn=loss_fn, _optimizer=optimizer)

    n_epochs = 100
    training_losses = []
    validation_losses = []
    print(model.state_dict())

    for epoch in range(n_epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_step(_x=x_batch, _y=y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses = []
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()
                y_hat = model(x_val)
                val_loss = loss_fn(y_val, y_hat).item()
                val_losses.append(val_loss)
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        print(
            f"[Epoch: {epoch + 1: 05d} Training loss : {training_loss:.3f}\t Validation loss : {validation_loss: .3f}]")

    print(model.state_dict())
