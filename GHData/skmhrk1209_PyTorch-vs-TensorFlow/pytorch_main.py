import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )
        self.fc1 = nn.Linear(
            in_features=7 * 7 * 64,
            out_features=1024
        )
        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=10
        )

    def forward(self, inputs):

        inputs = self.conv1(inputs)
        inputs = nn.functional.relu(inputs)
        inputs = self.pool1(inputs)
        inputs = self.conv2(inputs)
        inputs = nn.functional.relu(inputs)
        inputs = self.pool2(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.fc1(inputs)
        inputs = nn.functional.relu(inputs)
        inputs = self.fc2(inputs)
        inputs = nn.functional.log_softmax(inputs, dim=1)

        return inputs


def train(model, device, data_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("Train: Epoch: {}[{}/{}], Loss: {:.6f}".format(
                    epoch, step * len(images), len(data_loader.dataset), loss.item()
                ))
                if step % 1000 == 0:
                    torch.save(model.state_dict(), "mnist_convnet_model.pt")


def test(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss += nn.functional.nll_loss(logits, labels, reduction="sum").item()
            predictions = logits.argmax(1)
            correct += predictions.eq(labels).sum().item()
    print("Test: Average loss: {:.6f}, Accuracy: {:.2f}%".format(
        loss / len(data_loader.dataset), correct / len(data_loader.dataset) * 100.
    ))


if __name__ == "__main__":

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=100,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=100,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    device = torch.device("cuda")

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters())

    begin = time.time()
    train(model, device, train_loader, optimizer, 10)
    test(model, device, test_loader)
    end = time.time()

    print("elapsed_time: {}s".format(end - begin))
