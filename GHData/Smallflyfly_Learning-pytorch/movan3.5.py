import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

BATH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATH_SIZE,
    shuffle = True,
    num_workers = 2,
)
for epoch in range(3):
    for step, (bath_x, bath_y) in enumerate(loader):
        print('Epoch: ', epoch, '| Step: ', step, '| bath x: ', 
        bath_x.numpy(), '|bath_y: ', bath_y.numpy())
