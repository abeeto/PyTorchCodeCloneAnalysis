import torch
import torch.nn as nn
import numpy as np
import os
import poptorch

input_size = 16
hidden_size = 16
training_batch_size = 1
time_seq = 50
training_ipu_step_size = 1
replication_factor = 1

training_combined_batch_size = training_batch_size * training_ipu_step_size * replication_factor

x = torch.randn(time_seq, training_combined_batch_size, input_size)
y = torch.zeros_like(x)


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

  def forward(self, x):
    y, _ = self.lstm(x)
    return y


model = Model()

opts = poptorch.Options()
opts.deviceIterations(training_ipu_step_size)
opts.replicationFactor(replication_factor)

train_model = poptorch.trainingModel(model, options=opts,
                                     loss=nn.MSELoss(reduction="mean"))

for i in range(1):
  result = train_model(x, y)
  print(result)
