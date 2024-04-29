import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import confusion_matrix

##
## model
##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

##
## test main function
##
def test(model, device, dataset):

    testloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)

    data, labels_ground_truth = testloader.__iter__().next()
    
    labels_ground_truth = labels_ground_truth.numpy().copy()

    _pred = model(data).numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    result = confusion_matrix(labels_ground_truth, labels_pred)

    print(result)

    
def main():

    #
    # test dataset
    #
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('data', train=False, transform=transform)


    #
    # model
    #
    modelname = "mnist_cnn.pt"
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(modelname))

    #
    # testing
    #
    with torch.no_grad():
        model.eval()
        test(model, device, dataset)


if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
