import torch
import torchvision
from MyDataset import Dataset, TestDataset, ToTensor
import Models
import numpy as np
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F


def init_datasets():
    train_x = np.loadtxt(sys.argv[1], dtype=float)
    train_y = np.loadtxt(sys.argv[2], dtype=int)
    # Normalize
    train_x /= 255
    mean_x = np.mean(train_x)
    std_x = np.std(train_x)
    # Z score
    train_x = (train_x - mean_x) / std_x
    input_size_x = train_x.shape[1]
    n_samples = len(train_x)
    # Create test set and validation set
    per80 = int(0.8 * n_samples)
    training_l, validation_l = train_x[:per80, :], train_x[per80:, :]
    labels_training, labels_validation = train_y[:per80], train_y[per80:]
    # transforms
    transforms = torchvision.transforms.Compose([ToTensor()])
    # Call dataset for training
    training_set = Dataset(training_l, labels_training,  transforms)
    train_loader_l = DataLoader(dataset=training_set, batch_size=64, shuffle=True)
    # Call dataset for validation
    validation_set = Dataset(validation_l, labels_validation, transforms)
    validation_loader_l = DataLoader(dataset=validation_set, batch_size=1, shuffle=False)
    return input_size_x, mean_x, std_x, train_loader_l, validation_loader_l


def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    for data, label_data in train_loader:
        model.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label_data.type(torch.int64))
        # backward process
        loss.backward()
        # update optimizer
        model.optimizer.step()
        train_loss += loss
        # get the index of the max log probability
        pred = output.max(1, keepdim=True)[1]
        # Sum all the correct classifications
        correct += pred.eq(label_data.view_as(pred)).sum().item()
    train_loss /= (len(train_loader.dataset) / 64)
    """
     print('Train set, epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    """


def test(epoch, loader):
    # The model behave differently during training and inference (evaluating) time.
    model.eval()
    test_loss = 0
    correct = 0
    # using torch.no_grad() in pair with model.eval() turn off gradients computation
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            # Sum up batch loss
            # reduction='mean': the reduced loss will be the average of all entries
            test_loss += F.nll_loss(output, target.type(torch.int64), reduction='sum').item()
            # get the index of the max log probability
            pred = output.max(1, keepdim=True)[1]
            # Sum all the correct classifications
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        """
        print('Test set, epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch, test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        """


def classification(test_samples_loader):
    test_classification = []
    model.eval()
    with torch.no_grad():
        for sample in test_samples_loader:
            output = model(sample.float())
            pred = output.max(1, keepdim=True)[1]
            test_classification.append(pred.item())
    with open('test_y', 'w') as f:
        for y_hat in test_classification:
            f.write("%s\n" % y_hat)


if __name__ == '__main__':
    input_size, mean, std, train_loader, validation_loader = init_datasets()
    # Call dataset of FashionMNIST for test the model
    transforms_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((mean,), (std,))])
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(
        root="../fashion_data", download=True, train=False, transform=transforms_test), shuffle=False)
    # Select the model - A,B,C,D,E,F
    model = Models.ModelD(input_size)
    for epoch in range(1, 1 + 10):
        train(epoch)
        test(epoch, validation_loader)
    print("FashionMNIST test")
    test(1, test_loader)
    # Classify the test with the best model
    test_x = np.loadtxt(sys.argv[3], dtype=float)
    # Normalize
    test_x /= 255
    # Z score
    test_x = (test_x - mean) / std
    test_classify_set = TestDataset(test_x)
    test_classify_loader = DataLoader(dataset=test_classify_set, shuffle=False)
    classification(test_classify_loader)
