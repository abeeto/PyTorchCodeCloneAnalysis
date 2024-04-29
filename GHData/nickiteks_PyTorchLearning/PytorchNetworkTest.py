import torch
from net import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

x = torch.FloatTensor([0.5])
x.unsqueeze_(1)
pred = model(x)
print(pred)