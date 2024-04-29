from torch import nn
import collections


class FC(nn.Module):
    def __init__(self, input_nums, out_nums, dropout=None):
        super(FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_nums, out_nums),
            nn.BatchNorm1d(out_nums),
            nn.ReLU()
        )
        if dropout is not None:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_nums, out_nums),
                nn.BatchNorm1d(out_nums),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.fc(x)


class OverFitFC(nn.Module):
    def __init__(self, nums, drops=None):
        super(OverFitFC, self).__init__()
        _len = len(nums)
        self.networks = collections.OrderedDict()
        if drops is not None:
            for i in range(_len - 1):
                self.networks['fc' + str(i)] = FC(nums[i], nums[i + 1], drops[i])
        else:
            for i in range(_len - 1):
                self.networks['fc' + str(i)] = FC(nums[i], nums[i + 1])
        self.network = nn.Sequential(self.networks)

    def forward(self, x):
        return self.network(x)
