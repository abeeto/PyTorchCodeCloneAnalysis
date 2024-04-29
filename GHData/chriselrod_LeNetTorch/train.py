
import torch
import torchvision
import time
import os

path = os.path.dirname(os.path.realpath(__file__))
USE_CUDA = torch.cuda.is_available()
print('Use cuda:', USE_CUDA)
device = torch.device("cuda:2" if USE_CUDA else "cpu")
NUM_THREADS = len(os.sched_getaffinity(0))
BATCH_SIZE = 2048 if USE_CUDA else 16*NUM_THREADS

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 6, 5)
        torch.nn.init.xavier_uniform_(self.conv0.weight)
        self.r0 = torch.nn.ReLU()
        self.mp0 = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(6, 16, 5)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.r1 = torch.nn.ReLU()
        self.mp1 = torch.nn.MaxPool2d(2)
        self.d0 = torch.nn.Linear(256, 120)
        torch.nn.init.xavier_normal_(self.d0.weight)
        self.r2 = torch.nn.ReLU()
        self.d1 = torch.nn.Linear(120, 84)
        torch.nn.init.xavier_normal_(self.d1.weight)
        self.r3 = torch.nn.ReLU()
        self.d2 = torch.nn.Linear(84, 10)
        torch.nn.init.xavier_normal_(self.d2.weight)
        self.out = torch.nn.LogSoftmax(1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.r0(x)
        x = self.mp0(x)
        x = self.conv1(x)
        x = self.r1(x)
        x = self.mp1(x)
        x = self.d0(x.view(x.shape[0], -1))
        x = self.r2(x)
        x = self.d1(x)
        x = self.r3(x)
        x = self.d2(x)
        x = self.out(x)
        return x

def report_error(model, data):
    correct = 0
    count = 0
    for (x, y) in data:
        correct += sum(torch.argmax(model(x.to(device)),dim=1) == y.to(device))
        count += len(y)
    print('Accuracy: {:.4f}'.format(correct / count))
    
    
def train():
    epochs = 10
    train = torchvision.datasets.MNIST(path, train = True, download = True,
        transform = torchvision.transforms.ToTensor())
    train.data.to(device)
    train.targets.to(device)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size = BATCH_SIZE, shuffle = True,
        num_workers = NUM_THREADS//2, pin_memory = True)
    test = torchvision.datasets.MNIST(path, train = False, download = True,
        transform = torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test, batch_size = BATCH_SIZE, shuffle = True,
        num_workers = NUM_THREADS//2, pin_memory = True)

    model = LeNet().to(device)
    adam = torch.optim.Adam(model.parameters(), lr = 3e-4)

    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(2):
        t_start = time.time()
        for _ in range(epochs):
            for (x, y) in train_loader:
                adam.zero_grad()
                loss_fn(model(x.to(device)), y.to(device)).backward()
                adam.step()
        print('Took: {:.2f}'.format(time.time() - t_start))
        report_error(model, test_loader)
    
if __name__ == '__main__':
    train()
    
