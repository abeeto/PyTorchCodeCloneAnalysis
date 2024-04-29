import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

# Constants
IMAGE_SIZE = 28 * 28
BATCH_SIZE = 200
NEPOCHS = 10
PERCENT = 0.8


class Model_A(nn.Module):
    '''
    Model A - Neural Network with two hidden layers.
    first layer - size 100 with ReLU Activation.
    second layer - size 50 with ReLU Activation.
    '''

    def __init__(self, ):
        super(Model_A, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), -1)


class Model_B(nn.Module):
    '''
    Model B - Neural Network with two hidden layers.
    first layer - size 100 with ReLU Activation.
    second layer - size 50 with ReLU Activation.
    '''

    def __init__(self):
        super(Model_B, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), -1)


class Model_C(nn.Module):
    '''
    Model C - Neural Network Dropout on Model A.
    first layer - size 100 with ReLU Activation then Dropout.
    second layer - size 50 with ReLU Activation then Dropout.
    '''

    def __init__(self):
        super(Model_C, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fd1 = torch.nn.Dropout(p=0.001)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))
        x = self.fd1(x)
        x = F.relu(self.fc1(x))
        x = self.fd1(x)
        return F.log_softmax(x, -1)


class Model_D(nn.Module):
    '''
        Model D - Adding Batch Normalization to Model A.
        first layer - size 100 with Batch Norm.
        second layer - size 50 with Batch Norm.
        '''

    def __init__(self):
        super(Model_D, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = torch.nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.batch_x1 = torch.nn.BatchNorm1d(100)
        self.batch_x2 = torch.nn.BatchNorm1d(50)
        self.batch_x3 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.fc0(x)
        x = F.relu(self.batch_x1(x))
        x = self.fc1(x)
        x = F.relu(self.batch_x2(x))
        x = self.fc2(x)
        return F.log_softmax(x, -1)


class Model_E(nn.Module):
    '''
        Model E - Neural Network with five hidden layers.
        first layer - size 128 with ReLU Activation.
        second layer - size 64 with ReLU Activation.
        third layer - size 10 with ReLU Activation.
        fourth layer - size 10 with ReLU Activation.
        fifth layer - size 10 with ReLU Activation.
        '''

    def __init__(self):
        super(Model_E, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = torch.nn.Linear(IMAGE_SIZE, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))  # Hidden layer 1
        x = F.relu(self.fc1(x))  # Hidden layer 2
        x = F.relu(self.fc2(x))  # Hidden layer 3
        x = F.relu(self.fc3(x))  # Hidden layer 4
        x = F.relu(self.fc4(x))  # Hidden layer 5
        return F.log_softmax(self.fc5(x), -1)


class Model_F(nn.Module):
    '''
        Model F - Neural Network with two hidden layers.
        first layer - size 128 with Sigmoid Activation.
        second layer - size 64 with Sigmoid Activation.
        third layer - size 10 with Sigmoid Activation.
        fourth layer - size 10 with Sigmoid Activation.
        fifth layer - size 10 with Sigmoid Activation.
        '''

    def __init__(self):
        super(Model_F, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = torch.nn.Linear(IMAGE_SIZE, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = torch.sigmoid(self.fc0(x))  # Hidden layer 1
        x = torch.sigmoid(self.fc1(x))  # Hidden layer 2
        x = torch.sigmoid(self.fc2(x))  # Hidden layer 3
        x = torch.sigmoid(self.fc3(x))  # Hidden layer 4
        x = torch.sigmoid(self.fc4(x))  # Hidden layer 5
        return F.log_softmax(self.fc5(x), -1)


def train_model(model, optimizer, criterion, train_x, train_y, val_x, val_y, name_of_model, batch_size):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    criterion - the criterion (loss function)
    nepochs - number of training epochs
    train_x - all images from the trainset
    train_y - all labels from the trainset
    val_x - all images from the validation set
    val_y - all labels from the validation set
    '''
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    train_length = len(train_x)
    val_length = len(val_x)
    print(f"Running now MODEL {name_of_model}:")
    for e in range(NEPOCHS):
        running_loss = 0
        running_val_loss = 0
        running_train_acc = 0
        running_val_acc = 0
        # training_set, lables_set = shuffle(train_x.dataset, train_y.dataset)
        for batch_idx, (image, label) in enumerate(zip(train_x, train_y)):
            # Training pass
            model.train()  # set model in train mode
            optimizer.zero_grad()
            model_out = model(image)
            a = torch.reshape(label, (batch_size,))
            loss = criterion(model_out, a.long())
            # loss = F.nll_loss(model_out, a.long())
            loss.backward()
            # one gradient descent step
            optimizer.step()
            running_loss += loss.item()
            temp = model_out.detach().numpy()
            # label = label.detach().numpy()
            for i in range(0, len(temp)):
                y_hat = np.argmax(temp[i])
                if y_hat == label[i]:
                    running_train_acc += 1
        # Validation
        else:
            val_loss = 0
            # Evaluate model on validation at the end of each epoch.
            with torch.no_grad():
                for image, label in zip(val_x, val_y):
                    # Validation pass
                    model_out = model(image)
                    a = torch.reshape(label, (batch_size,))
                    val_loss = criterion(model_out, a.long())
                    temp = model_out.detach().numpy()
                    # label = label.detach().numpy()
                    for i in range(0, len(temp)):
                        y_hat = np.argmax(temp[i])
                        if y_hat == label[i]:
                            running_val_acc += 1
                    running_val_loss += val_loss
            # Track train loss and validation loss
            train_losses.append(running_loss / (train_length * batch_size))
            val_losses.append(running_val_loss / (val_length * batch_size))
            # Track train acc and validation acc
            train_acc.append(running_train_acc / (train_length * batch_size))
            val_acc.append(running_val_acc / (val_length * batch_size))
            print("Epoch: {}/{}.. ".format(e + 1, NEPOCHS),
                  "Training Loss: {:.3f}.. ".format(running_loss / (train_length * batch_size)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / (val_length * batch_size)))
            print("Epoch: {}/{}.. ".format(e + 1, NEPOCHS),
                  "Training Acc: {:.3f}.. ".format(running_train_acc / (train_length * batch_size)),
                  "Validation Acc: {:.3f}.. ".format(running_val_acc / (val_length * batch_size)))
    return train_losses, val_losses, train_acc, val_acc


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(str(model) + '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_test_loader(mean, std):
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=my_transforms),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return test_loader


def plot(train_losses, val_losses, train_acc, val_acc, nepochs, model_str):
    # plot train and validation loss as a function of #epochs
    epochs = [*range(1, nepochs + 1)]
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss - ' + model_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy - ' + model_str)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def split_data(x_data, percent):
    test_size = int(percent * x_data.shape[0])
    train_data = x_data[:test_size]
    test_data = x_data[test_size:]
    return train_data, test_data


def predict(model, test_x):
    model.eval()
    labels = []
    with torch.no_grad():
        for x in test_x:
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            labels.append(pred)
    with open('test_y', 'w') as fp:
        fp.write("\n".join(str(item.item()) for item in labels))


def main():
    # Get data input and create numpy array
    train_data_x, train_data_y, test_x = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2]), np.loadtxt(sys.argv[3])
    # Normalize all data + test_x
    train_data_x = train_data_x / 255.0
    test_x = test_x / 255.0
    std = train_data_x.std()
    mean = train_data_x.mean()
    train_data_x = train_data_x - mean
    train_data_x = train_data_x / std
    test_x = test_x - mean
    test_x = test_x / std
    # Create tensors from the data
    new_data = np.column_stack((train_data_x, train_data_y))
    np.random.shuffle(new_data)
    temptrain_x, tempval_x = split_data(new_data, PERCENT)
    val_x = tempval_x[:, :IMAGE_SIZE]
    val_y = tempval_x[:, IMAGE_SIZE]
    train_x = temptrain_x[:, :IMAGE_SIZE]
    train_y = temptrain_x[:, IMAGE_SIZE]
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_x = DataLoader(dataset=train_x, batch_size=BATCH_SIZE)
    val_x = DataLoader(dataset=val_x, batch_size=BATCH_SIZE)
    train_y = DataLoader(dataset=train_y, batch_size=BATCH_SIZE)
    val_y = DataLoader(dataset=val_y, batch_size=BATCH_SIZE)
    # Choose loss function
    criterion = nn.NLLLoss()
    # Download test and normalize according to our mean and std
    test_loader = get_test_loader(mean, std)

    # ================================================== THE MODELS ==================================================

    # # ================================================== MODEL A ==================================================
    # # Build the architecture of the network
    # model_a = Model_A()
    # model_str = 'A'
    # # Define the optimizer function and the value of the learning rate
    # lr = 0.1
    # # SGD optimizer
    # optimizer = optim.SGD(model_a.parameters(), lr=lr)
    # # Train
    # train_losses, val_losses, train_acc, val_acc = train_model(model_a, optimizer, criterion, train_x, train_y,
    #                                                            val_x, val_y, name_of_model='A', batch_size=BATCH_SIZE)
    # # plot train and validation loss as a function of #epochs
    # plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # # Test PyTorch test-set
    # test(model_a, test_loader)
    #
    # # ================================================== MODEL B ==================================================
    # # Build the architecture of the network
    # model_b = Model_B()
    # model_str = 'B'
    # # Define the optimizer function and the value of the learning rate
    # lr = 0.001
    # # SGD optimizer
    # optimizer = optim.Adam(model_b.parameters(), lr=lr)
    # # Train
    # train_losses, val_losses, train_acc, val_acc = train_model(model_b, optimizer, criterion, train_x, train_y,
    #                                                            val_x, val_y, name_of_model='B', batch_size=BATCH_SIZE)
    # # plot train and validation loss as a function of #epochs
    # plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # # Test PyTorch test-set
    # test(model_b, test_loader)
    #
    # # ================================================== MODEL C ==================================================
    # # Build the architecture of the network
    # model_c = Model_C()
    # model_str = 'C'
    # # Define the optimizer function and the value of the learning rate
    # lr = 0.001
    # # SGD optimizer
    # optimizer = optim.Adagrad(model_c.parameters(), lr=lr)
    # # Train
    # train_losses, val_losses, train_acc, val_acc = train_model(model_c, optimizer, criterion, train_x, train_y,
    #                                                            val_x, val_y, name_of_model='C', batch_size=BATCH_SIZE)
    # # plot train and validation loss as a function of #epochs
    # plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # # Test PyTorch test-set
    # test(model_c, test_loader)

    # ================================================== MODEL D ==================================================
    # Build the architecture of the network
    model_d = Model_D()
    model_str = 'D'
    # Define the optimizer function and the value of the learning rate
    lr = 0.01
    # SGD optimizer
    optimizer = optim.Adagrad(model_d.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_d, optimizer, criterion, train_x, train_y,
                                                               val_x, val_y, name_of_model='D', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # Test PyTorch test-set
    test(model_d, test_loader)
    # Create file with the predicts of the user test_x
    predict(model_d, torch.from_numpy(test_x))

    # # ================================================== MODEL E ==================================================
    # # Build the architecture of the network
    # model_e = Model_E()
    # model_str = 'E'
    # # Define the optimizer function and the value of the learning rate
    # lr = 0.001
    # # SGD optimizer
    # optimizer = optim.Adam(model_e.parameters(), lr=lr)
    # # Train
    # train_losses, val_losses, train_acc, val_acc = train_model(model_e, optimizer, criterion, train_x, train_y,
    #                                                            val_x, val_y, name_of_model='E', batch_size=BATCH_SIZE)
    # # plot train and validation loss as a function of #epochs
    # plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # # Test PyTorch test-set
    # test(model_e, test_loader)
    #
    # # ================================================== MODEL E ==================================================
    # # Build the architecture of the network
    # model_f = Model_F()
    # model_str = 'F'
    # # Define the optimizer function and the value of the learning rate
    # lr = 0.001
    # # SGD optimizer
    # optimizer = optim.Adam(model_f.parameters(), lr=lr)
    # # Train
    # train_losses, val_losses, train_acc, val_acc = train_model(model_f, optimizer, criterion, train_x, train_y,
    #                                                            val_x, val_y, name_of_model='F', batch_size=BATCH_SIZE)
    # # plot train and validation loss as a function of #epochs
    # plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS, model_str)
    # # Test PyTorch test-set
    # test(model_f, test_loader)


if __name__ == '__main__':
    main()
