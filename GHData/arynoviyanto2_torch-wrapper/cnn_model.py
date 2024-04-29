import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnModel(nn.Module):

    def __init__(self, num_targets, dataset_name = ''):
        super(CnnModel, self).__init__()

        self.dataset_name = dataset_name
        self.num_targets = num_targets

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8*8*128, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_targets)

    def forward(self, batch_features): 
        # batch_features: batch_size x 64 x 64 x 3
        # Layer 1
        batch_features = self.maxPool1(self.bn1(self.conv1(batch_features))) # batch_size x 32 x 32 x 32
        batch_features = F.relu(batch_features)

        # pdb.set_trace()

        # Layer 2
        batch_features = self.maxPool2(self.bn2(self.conv2(batch_features))) # batch_size x 16 x 16 x 64
        batch_features = F.relu(batch_features)

        # Layer 3
        batch_features = self.maxPool3(self.bn3(self.conv3(batch_features))) # batch_size x 8 x 8 x 128
        batch_features = F.relu(batch_features)

        batch_features = batch_features.view(-1, 8*8*128) # batch_size x (8 * 8 * 128)

        # Layer 3
        batch_features = self.fcbn1(self.fc1(batch_features))
        batch_features = F.relu(batch_features)

        # Layer 4
        batch_features = self.dropout(batch_features)

        # Layer 5
        batch_features = self.fc2(batch_features)

        # Output
        return F.log_softmax(batch_features, dim=1)

    def get_name(self):
        return 'cnn_' + self.dataset_name