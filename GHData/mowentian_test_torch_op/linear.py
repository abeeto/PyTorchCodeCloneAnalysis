import numpy as np
import os

# from ctypes import *
#
# for pp in [
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/poplar-ubuntu_18_04-1.2.100+9677-c27b85b309/lib/',
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/popart-ubuntu_18_04-1.2.100-63af2bbaea/lib',
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/gc_drivers-ubuntu_18_04-1.0.44+1604-325648412e/lib',
#            ]:
#   for l in os.listdir(pp):
#     if '.so' in l and 'lib' in l:
#       print(os.path.join(pp, l))
#       cdll.LoadLibrary(os.path.join(pp, l))
#
# for pp in [
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/poplar-ubuntu_18_04-1.2.100+9677-c27b85b309/lib/',
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/popart-ubuntu_18_04-1.2.100-63af2bbaea/lib',
#   '/data/sdk/poplar_sdk-ubuntu_18_04-1.2.0+131-495c1aa368/gc_drivers-ubuntu_18_04-1.0.44+1604-325648412e/lib',
#            ]:
#   for l in os.listdir(pp):
#     if '.so' in l and 'lib' not in l:
#       print(os.path.join(pp, l))
#       cdll.LoadLibrary(os.path.join(pp, l))

import torch
import torch.nn as nn
import poptorch

in_dim = 2048
out_dim = 2048
training_batch_size = 2
gradient_accumulation = 4
replication_factor = 2
training_ipu_step_size = 1

training_combined_batch_size = training_batch_size * training_ipu_step_size * gradient_accumulation * replication_factor

torch.manual_seed(1024)
x = torch.randn(training_combined_batch_size, in_dim)
y = torch.zeros_like(x)


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

  def forward(self, x):
    y = self.linear(x)
    return y


model = Model()

opts = poptorch.Options()
opts.deviceIterations(training_ipu_step_size)
opts.replicationFactor(replication_factor)
opts.anchorMode(poptorch.AnchorMode.All)
opts.Training.gradientAccumulation(gradient_accumulation)

print('torch model(x) --------------------------------------------------------')
torch_out = model(x)
print(torch_out)
print('torch model(x) --------------------------------------------------------')
train_model = poptorch.trainingModel(model, options=opts,
                                     loss=nn.MSELoss(reduction="mean"))

for i in range(1):
  poptorch_out, loss = train_model(x, y)
  print('pptorch model(x) --------------------------------------------------------')
  print(poptorch_out)
  print(poptorch_out.shape)
  print('pptorch model(x) --------------------------------------------------------')

for i in range(poptorch_out.shape[0]):
  print(f'{i}---------------------------------------------------------------------')
  print(torch_out[i])
  print(poptorch_out[i])
