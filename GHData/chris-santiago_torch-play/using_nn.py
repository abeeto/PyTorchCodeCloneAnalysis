import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from get_mnist import load_data

MNIST = load_data().clean(binarize=False)


class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
        self.loss_func = nn.CrossEntropyLoss()  # includes log softmax activation

    def forward(self, xb):
        return self.lin(xb.float())


class MnistTrainer:
    def __init__(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, x_train, y_train, bs):
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=bs)
        for epoch in tqdm.tqdm(range(self.epochs)):
            for xb, yb in dataloader:
                pred = self.model.forward(xb)
                loss = self.model.loss_func(pred, yb)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

    def predict(self, x_valid):
        with torch.no_grad():
            self.model.eval()
            return self.model.forward(x_valid)


def accuracy(preds, actual):
    return (preds.argmax(1) == actual).float().mean()


def main():
    x_train, y_train, x_valid, y_valid = map(torch.tensor, tuple(MNIST))
    model = MnistLogistic()
    learner = MnistTrainer(model, epochs=20, lr=0.1)
    learner.fit(x_train, y_train.long(), 256)
    fitted = learner.predict(x_train)
    fitted_loss = model.loss_func(fitted, y_train.long())
    fitted_acc = accuracy(fitted, y_train.long())
    print(f'Training Loss: {fitted_loss}.  Training Accuracy: {fitted_acc}.')
    pred = learner.predict(x_valid)
    loss = model.loss_func(pred, y_valid.long())
    acc = accuracy(pred, y_valid.long())
    print(f'Validation Loss: {loss}.  Validation Accuracy: {acc}.')


if __name__ == '__main__':
    main()
