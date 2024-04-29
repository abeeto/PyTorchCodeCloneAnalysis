from typing import List

import numpy as np
import torch

from torchazoic.train import train
from torchazoic.model import Model
from torchazoic.layers import Linear, Tanh
from torchazoic.optimizers import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

inputs = torch.tensor([
    binary_encode(x)
    for x in range(101, 1024)
], dtype=torch.float)

targets = torch.tensor([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = Model([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    x_in = torch.tensor(binary_encode(x),dtype=torch.float)
    predicted = net.forward(x_in).numpy()
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])
