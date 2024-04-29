################# load packages #################
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

################# preprocess data #################
def preprocess_data():

    ######### set data transform ###########
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])


    ######### download data ###########
    data_train = datasets.MNIST(root="MNIST/", transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root="MNIST/", transform=transform, train=False)


    ######### data loader ###########
    batch_size = 128
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    return data_loader_train, data_loader_test


################# LeNet module #################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

################# train #################
def train(train_loader, model, optimizer, cost, epoch):

    print("Start training:")

    ########### epoch ##########
    for i in range(epoch):
        train_correct = 0
        total_cnt = 0

        ############## batch #############
        for batch_idx, (data, target) in enumerate(train_loader):

            ############ get data and target #########
            data, target = Variable(data), Variable(target)

            ############ optimizer ############
            optimizer.zero_grad()

            ############ get model output ############
            output = model(data)

            ############ get predict label ############
            _, pred = torch.max(output.data, 1)

            ############ loss ############
            loss = cost(output, target)
            loss.backward()

            ############ optimizer ############
            optimizer.step()

            ############ result ############
            total_cnt += data.data.size()[0]
            train_correct += torch.sum(pred == target.data)

            ############ show train result ############
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(data_loader_train):
                print("epoch: {}, batch_index: {}, train loss: {:.6f}, train correct: {:.2f}%".format(
                    i, batch_idx+1, loss, 100*train_correct/total_cnt))

    print("Training is over!")


################# test #################
def test(test_loader, model, cost):

    print("Start testing:")

    ############ batch ############
    for batch_idx, (data, target) in enumerate(test_loader):

        ############ get data and target ############
        data, target = Variable(data), Variable(target)

        ############ get model output ############
        output = model(data)

        ############ get predict label ############
        _,pred = torch.max(output.data, 1)

        ############ loss ############
        loss = cost(output, target)

        ############ accuracy ############
        test_correct = torch.sum(pred == target.data)

        print("batch_index: {}, test loss: {:.6f}, test correct: {:.2f}%".format(
                batch_idx + 1, loss.item(), 100*test_correct/data.data.size()[0]))

    print("Testing is over!")


################# main #################
if __name__ == '__main__':

    ################# get train and test data #################
    data_loader_train, data_loader_test = preprocess_data()

    ################# LeNet module #################
    model = LeNet()
    
    ########## 这里打印的model结构是LeNet 的 __init__ 下的结构 #########
    print("LeNet model is as follows:")
    print(model)

    ################# optimizer and loss #################
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cost = nn.CrossEntropyLoss()

    ################# train #################
    epoch = 20
    train(data_loader_train, model, optimizer, cost, epoch)

    ################# test #################
    test(data_loader_test, model, cost)
