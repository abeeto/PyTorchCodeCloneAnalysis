import torch
import torchvision


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)  # in_size = batch_size
        x1 = torch.nn.functional.relu(self.mp(self.conv1(x)))
        x2 = torch.nn.functional.relu(self.mp(self.conv2(x1)))
        x3 = x2.view(in_size, -1)  # flatten the tensor
        x4 = self.fc(x3)
        return torch.nn.functional.log_softmax(x4, dim=1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                          batch_idx * len(data),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader),
                                                                          loss.item()))


def test():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss = test_loss + torch.nn.functional.nll_loss(output, target, reduction='sum').data
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss = test_loss / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                     correct,
                                                                                     len(test_loader.dataset),
                                                                                     100. * correct / len(test_loader.dataset)))




if __name__ == '__main__':
    # Training settings
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training MNIST Model on {device}')
    print(f'\n{"=" * 44}')

    # MNIST Dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/mnist_data/',
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist_data/',
                                              train=False,
                                              transform=torchvision.transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    model = Net()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1,3):
        train(epoch)
        test()