from torch import nn
from torch.nn import functional


class Classifier(nn.Module):
    """Custom classifier."""

    def __init__(self, in_features=25088, hidden_features=4096, out_features=102, drop_prob=0.5):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.drop(functional.relu(self.fc1(x)))
        x = self.fc2(x)
        x = functional.log_softmax(x, dim=1)
        return x
