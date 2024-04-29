import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

def test_net(model, criterion, testloader):
        # test the net
        test_loss = 0
        correct = 0
        log_interval = 10
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            net_out = model(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct += batch_labels.sum()
            if batch_idx % log_interval == 0:
                    print('Testing: [{}/{} ({:.0f}%)]\t'.format(batch_idx * len(data),
                                                                                   len(testloader.dataset),
                                                                                   100. * batch_idx / len(testloader)))
        test_loss /= len(testloader.dataset)
        acc = 100. * float(correct) / len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset), acc))


class CNN(nn.Module):
    def __init__(self, dropout_rate):
        super(CNN, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dr1 = nn.Dropout(p=dropout_rate/2)
        self.fc1 = nn.Linear(9216, 128)
        self.dr2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(-1,9216)
        x = self.dr1(x)
        x = F.relu(self.fc1(x))
        x = self.dr2(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def train_net(self, criterion, optimizer, trainloader, epochs):
        log_interval = 10
        for epoch in range(epochs):
            # Freeze quantizer parameters after around 40% of epochs
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)

                net_out = self(data)
                loss = criterion(net_out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                   len(trainloader.dataset),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                loss.data.item()))

if __name__ == "__main__":
    flags = {"train": True,
             "test": True}
    
    batch_size = 128
    epochs=10
    dropout_rate = 0.5
    lr=0.01
    momentum=0.9

    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(dropout_rate)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    #Different Loss Function due to no Softmax Activation (which is not supported on quantized models in PyTorch)
    criterion = nn.CrossEntropyLoss()

    traced_path = 'cnn_net_quantized_traced_ptq.pt'

    if flags["train"]:
        model.train(True)
        #Train in normal manner, quantize after training is done
        model.train_net(criterion, optimizer, trainloader, epochs)
        for (data,target) in testloader:
            break

        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        #Insert fake nodes
        torch.quantization.prepare(model, inplace=True)
        
        #Feed data through the net, fake nodes keep track of activations for best quantization values
        test_net(model,criterion, testloader)

        torch.quantization.convert(model.eval(), inplace=True)

        torch.jit.save(torch.jit.trace(model,data), traced_path)

    if flags["test"]:
        traced_model = torch.jit.load(traced_path)
        traced_model.eval()
        with torch.no_grad():
            test_net(traced_model,criterion, testloader)