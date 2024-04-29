import torch
import math

from get_mnist import load_data

MNIST = load_data().clean()


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(batch, weights, bias):
    return log_softmax(batch.float() @ weights.float() + bias)


def nll(preds, actual):
    return -preds[range(actual.shape[0]), actual.argmax(1)].mean()


def binarize_labels(preds):
    labels = torch.zeros_like(preds)
    labels[range(preds.shape[0]), torch.argmax(preds, dim=1)] = 1
    return labels


def accuracy(preds, actual):
    preds = binarize_labels(preds)
    return (preds.argmax(1) == actual.argmax(1)).float().mean()


x_train, y_train, x_valid, y_valid = map(torch.tensor, tuple(MNIST))

# initialize weights and bias tensors
# initializing the weights here with Xavier initialisation (by multiplying with 1/sqrt(n)).
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# single batch
bs = 64
xb = x_train[0:bs]
yb = y_train[0:bs]
preds = model(xb, weights, bias)
loss = nll(preds, yb)
acc = accuracy(preds, yb)
print(f'Loss: {loss}. Accuracy: {acc}')

# training
lr = 0.5
epochs = 2
for epoch in range(epochs):
    for i in range((x_train.shape[0] - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb, weights, bias)
        loss = nll(pred, yb)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

# check metrics post training
preds = model(xb, weights, bias)
loss = nll(preds, yb)
acc = accuracy(preds, yb)
print(f'Loss: {loss}. Accuracy: {acc}')