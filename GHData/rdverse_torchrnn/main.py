import numpy as np
from Data import Data
import torch
from model import Net
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from testModel import Classifier


def check_accuracy(testloader):

    #Test the accuracy
    correct = 0
    total = 0
    #    df = pd.DataFrame(columns=['true', 'pred'])
    compare = np.array(0)
    with torch.no_grad():
        for data in testloader:
            feat, lab = data

            outputs = net(feat)

            # calculate total labels (should be 2000)
            total += labels.size(0)

            # Convert sigmoid output in the format of the labels
            predicted = np.array(
                [1 if x > 0 else 0 for x in outputs.flatten()])

            # Convert array into the format of the labels tensor
            predicted = torch.tensor(predicted).to(dtype=torch.float64)

            # Convert the predicted outputs to
            correct += (predicted == lab.flatten()).sum().item()
            compare = np.hstack((predicted.reshape(-1, 1), lab))
    print(compare.shape)
    compare = compare.astype(int)
    np.savetxt('results.txt', compare)
    print('Accuracy on test set: %d %%' % (100 * correct / total))

    #print('Train and test the accuracy of an SVC model with the same data')
    #Classifier(featuresTrain, featuresTest, labelsTrain, labelsTest)


if __name__ == '__main__':
    data = Data()
    featuresTrain, featuresTest, labelsTrain, labelsTest = [
        torch.from_numpy(item) for item in data.getData()
    ]

    EPOCHS = 3

    net = Net(input_size=512,
              hidden_size=512,
              num_layers=1,
              num_classes=2,
              sequence_length=512)

    # Loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Pytorch train and test sets
    labelsTrain = labelsTrain.reshape(-1, 1)
    labelsTest = labelsTest.reshape(-1, 1)

    train = TensorDataset(featuresTrain, labelsTrain)
    test = TensorDataset(featuresTest, labelsTest)

    # data loader
    batch_size = 100
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test, batch_size=2000, shuffle=False)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(
                outputs,
                labels.to(dtype=torch.float).reshape((batch_size, 1)))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print for every  epoch
                print('Epoch : %d, loss: %.3f' %
                      (epoch + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    check_accuracy(testloader)
