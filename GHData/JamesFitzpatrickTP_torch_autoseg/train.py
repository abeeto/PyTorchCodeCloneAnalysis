import torch
import network_builder as nb
import load_imgs
import numpy as np


def train(model, X, y, criterion, optimiser):
    y_hat = model(X)
    loss = criterion(y_hat, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return loss.item()


def learn(epochs, model, X, y, criterion, optimiser):
    for t in range(epochs):
        err = train(model, X, y, criterion, optimiser)
        print("Epoch: ", t, ", Loss: ", err)

#X, y  = load_imgs.torch_loader(load_imgs.train_dir, load_imgs.truth_dir, [0])
X = torch.tensor(np.zeros((1, 3, 1200, 1800))).float()
y = torch.tensor(np.zeros((1, 1, 1200, 1800))).float()
model = nb.NeuralNetwork([3, 10, 20, 40, 80], 3, 2, 2)
criterion = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.SGD(model.parameters(), lr=1e-9)

learn(10, model, X, y, criterion, optimiser)    

#torch.save(model, 'test.pt')
#model = torhc.load('test.pt')
