from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.unpool = nn.Upsample(scale_factor=7,mode='bilinear') #nn.MaxUnpool2d(2,stride=2)
        self.conv2_drop = nn.Dropout2d()

        self.fc = nn.Linear(20,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.unpool(x)

        # Global Avg pool
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #Test
    model = torch.load('lenet_model.pt')
    model_reduced = nn.Sequential(*list(model.children())[:-2])
    weights = model.fc.weight.data.cpu().numpy()

    data_num = 25 # Image whose CAM is to be generated

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            break

        image = data[data_num].data.cpu().numpy()
        output = model(data[data_num].unsqueeze(0))
        predict_class = output.data.cpu().numpy()
        print(np.argmax(predict_class))

        output = model_reduced(data[data_num].unsqueeze(0))
        feature_map = output.data.cpu().numpy()

        weight_mul = weights[np.argmax(predict_class)]
        weight_mul = np.reshape(weight_mul, (20,1))
        heat_map = np.zeros((feature_map.shape[2],feature_map.shape[3]))
        for i in range(20):
            heat_map += weight_mul[i][0] * feature_map[0][i]
        heat_map *= (255.0/heat_map.max())
        heat_map = np.reshape(heat_map,(140,140))
        img = cv2.resize(image[0], (140, 140), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Image", img)
        # cv2.waitKey(0)
        cv2.imshow("CAM", heat_map)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()