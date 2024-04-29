import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def compute_Loss(self, predicted_label, gold_label):
        return self.loss(predicted_label, gold_label)

    def load_model(self, save_path):
        self.load_state_dict(torch.load(save_path))

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
