import torch
import torch.nn.functional as F
from ProcessData import Data
from Net import Net
import torch.optim as optim
import matplotlib.pyplot as plt


DATA = Data()
EPOCHS = 5
LEARNING_RATE = 0.001
learningRate = 0.005


def main():
    net = Net()
    train(net)
    correct(net)
    output(net)


def output(net):
    image = 1
    while True:
        for data in DATA.test_set:
            X = data[0][image]
            print("predicted:",torch.argmax(net(X.view(-1, 784))[0]).item(), "      real:", data[1][image].item())
            plt.imshow(X.view(28, 28))
            plt.show()
        image += 1



def correct(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in DATA.test_set:
            X, y = data
            output = net(X.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print("Accuracy: ", round(correct/total, 3)*100, "%")


def train(net):
    global learningRate
    for epoch in range(EPOCHS):
        optimizer = optim.Adam(net.parameters(), lr=learningRate)
        learningRate *= 0.5
        for data in DATA.train_set:
            X, Y = data
            net.zero_grad()
            output = net(X.view(-1, 28*28))
            loss = F.nll_loss(output, Y)
            loss.backward()
            optimizer.step()
        print(loss)


if __name__ == '__main__':
    main()
