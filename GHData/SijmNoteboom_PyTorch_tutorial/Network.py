import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t


def get_num_correct(prediction, label):
        return prediction.argmax(dim=1).eq(label).sum().item()


torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)


for epoch in range(10):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # Get Batch
        images, labels = batch

        output = network(images)  # Pass Batch
        loss = F.cross_entropy(output, labels)  # Calculate Loss

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(output, labels)

    print(
        "epoch", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )

""" to vizualize 2 iterations in the epoch
batch = next(iter(data_loader))
images, labels = batch

output = network(images)
loss = F.cross_entropy(output, labels)

loss.backward()
optimizer.step()

print('loss1:', loss.item())
output = network(images)
loss = F.cross_entropy(output, labels)
print('loss2:', loss.item())
"""


"""
in order to render the random behaviour of the first iteration of the network
    network2 = Network2()
    pred2 = network(image.unsqueeze(0))
    print(pred2.argmax())
    print(F.softmax(pred2, dim=1).argmax(dim=1))
    print(pred2)
"""

total_correct/len(train_set)

torch.save({'state_dict': network.state_dict()}, 'checkpoint.pth.tar')  #r"\home\cdac-user\PycharmProjects\PyTorch_tutorial\model.pt")

len(train_set)

len(train_set.targets)   #labels

def get_all_preds(network, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds


prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get_all_preds(network, prediction_loader)

with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

train_preds.grad
train_preds.grad_fn

preds_correct = get_num_correct(train_preds, train_set.targets)

print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

stacked = torch.stack((train_set.targets, train_preds.argmax(dim=1)), dim=1)
print(stacked.shape)
print(stacked)

stacked[0].tolist()

cmt = torch.zeros(10,10, dtype=torch.int64)

for p in stacked:
    tl, pl = p.list()
    cmt[tl, pl] = cmt[tl, pl] + 1