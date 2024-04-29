import torch
from torchazoic.train import train
from torchazoic.model import Model
from torchazoic.layers import Linear, Tanh

inputs = torch.tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
], dtype=torch.float)

targets = torch.tensor([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

model = Model([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(model, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = model.forward(x)
    print(x, predicted, y)
