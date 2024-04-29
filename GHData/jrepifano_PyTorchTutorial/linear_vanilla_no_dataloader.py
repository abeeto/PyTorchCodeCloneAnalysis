import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # Set this flag to set your devices. For example if I set '6,7', then cuda:0 and cuda:1 in code will be cuda:6 and cuda:7 on hardware


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = torch.nn.Linear(20, 100)
        self.layer_2 = torch.nn.Linear(100, 50)
        self.layer_3 = torch.nn.Linear(50, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


def main():
    x, y = make_classification(n_samples=60000, n_features=20, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).float().to('cuda:0')
    x_test, y_test = torch.from_numpy(x_test).float().to('cuda:0'), torch.from_numpy(y_test).float().to('cuda:0')
    model = Model()
    no_epochs = 10
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to('cuda:0')
    train_loss = []
    test_loss = []
    for epoch in range(no_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(x_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        print('Epoch {}/{}: Training Loss: {:4f}'.format(epoch+1, no_epochs, loss.item()))
        train_loss.append(loss.item())
        model.eval()
        outputs = model.forward(x_test)
        loss = criterion(outputs, y_test.view(-1, 1))
        print('Test Loss: {:4f}'.format(loss.item()))
        test_loss.append(loss.item())
    plt.plot(np.arange(no_epochs), train_loss, label='Train Loss')
    plt.plot(np.arange(no_epochs), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # torch.save(model.state_dict(), os.getcwd()+'/saved_models/mlp.pt')    # To save model parameters, uncomment
    # model.load_state_dict(torch.load(os.getcwd()+'/saved_models/d.pt'))   # Use this to load them back in (obviously somewhere else)


if __name__ == '__main__':
    main()
