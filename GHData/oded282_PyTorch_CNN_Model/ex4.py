import torch.nn as nn
import torch.nn.functional as F
import torch
from gcommand_loader import GCommandLoader

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001


class ConvNet(nn.Module):
    """
        The algorithm flow:
        Four convolutoinal layers (7x7 size, padding size 2, stride size 1):
        Relu activation function, batch noramlization and MaxPool (size 2, stride size 2).
        Fully connected neural network using drop out technique.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 15, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(640, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = F.log_softmax(out, dim=1)
        return out


def load(dir, batch_size, shuffle):
    """
        This method responsible of loading and processing the data.
        We convert the audio files into sound wave pictures.
        Args:
            batch_size(int): the size of the batch.
            shuffle(callable): A function to shuffle the data.
    """
    dataset = GCommandLoader(dir)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, sampler=None)

    return loader


def train_model(model, test_loader, val_loader, train_loader):
    """
        This training set runs for 10 epochs.
        For each example in the training set we activate our neural network,
        calculate the loss using CrossEntropyLoss then use Adam optimizer.
        Then we check our model upon the validation set.
        After 10 epochs we run our model on the data test and write it to some file.
        Args:
            model(ConvNet): The convolutional neural network.
            test_loader(matrix): Matrix that contains all the proccessed data test.
            val_loader(array): array that contains all the right predictions.
            train_loader(matrix): Matrix that contains all the proccessed training test.
    """

    for epoch in range(num_epochs):
        # optimizer and loss func.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for i, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # Run the forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Back propagation and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test the model on the validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in val_loader:
                x, y = data
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print('Test Accuracy of the model, 10000 test images: {:.2f} %'.format((correct / total) * 100))

    model.eval()
    write_to_file(model, test_loader)


def write_to_file(model, test_loader):
    """
        This function test our trained model on the data test, and prints the reuslts to some file.
        Args:
            model(ConvNet): The convolutional neural network.
            test_loader(matrix): Matrix that contains all the proccessed data test.
    """
    f = open("test_y", "w")
    i = 0
    for x, y in test_loader:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        output = model(x)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        f.write(str(test_loader.dataset.spects[i][0].split("/")[4]) + ", " + str(pred[0].item()))
        f.write("\n")
        i += 1


def main():
    """
        The main function.
    """
    train_loader = load('./data/train', batch_size, True)
    val_loader = load('./data/valid', batch_size, True)
    test_loader = load('./data/test', 1, False)

    if torch.cuda.is_available():
        model = ConvNet().cuda()
    else:
        model = ConvNet()

    train_model(model, test_loader, val_loader, train_loader)


if __name__ == "__main__":
    main()
