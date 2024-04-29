from config import *
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(INIT_FEATURES, HIDDEN_LAYER_1_FEATURES)
        self.hidden_2 = nn.Linear(HIDDEN_LAYER_1_FEATURES, HIDDEN_LAYER_2_FEATURES)
        self.logits = nn.Linear(HIDDEN_LAYER_2_FEATURES, NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)
        nn.init.normal_(self.logits.weight, mean=0, std=0.2)

    def forward(self, input):
        first_hidden = self.input(input)
        hidden_2 = self.dropout(self.hidden_2(F.relu(first_hidden)))
        logits = self.dropout(self.logits(F.relu(hidden_2)))
        return logits
