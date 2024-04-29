import torch
import numpy as np
from model import FFNN

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    print(checkpoint["model"])
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

model = load_checkpoint("news_model.pth")

x = [[0] * 300]
x = np.array(x)
x = torch.from_numpy(x).to(torch.float32)
output = model(x)
_, predicted = torch.max(output, 1)
print(predicted.item())
