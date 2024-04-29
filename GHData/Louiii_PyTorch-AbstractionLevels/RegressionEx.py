import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from tqdm import tqdm
torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)
y = torch.sin(x)

x, y = Variable(x), Variable(y)

#######################################################################
#########                     MODEL 1:                        #########
#######################################################################
M = lambda n, m, r : nn.Parameter(torch.randn(n, m)/r)
V = lambda n : nn.Parameter(torch.zeros(n))

class ANN(nn.Module):
    def __init__(self, dims):
        super(ANN, self).__init__()
        self.Ws = [M(dims[i], dims[i+1], len(dims)*40) for i in range(len(dims)-1)]
        self.bs = [V(dims[i]) for i in range(1, len(dims))]

    def forward(self, x):
        next_layer = x
        for t, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = torch.matmul(next_layer, W) + b
            next_layer = F.relu(a) if t<len(self.bs)-1 else a
        return next_layer

    def parameters(self):
        return self.Ws + self.bs

#######################################################################
#########                     MODEL 2:                        #########
#######################################################################
# class ANN(nn.Module):
#     def __init__(self, dims):
#         super(ANN, self).__init__()
#         self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
#         self.prs = []
#         for l in self.layers:
#             self.prs += l.parameters()
#
#     def forward(self, x):
#         next_layer = x
#         for t, l in enumerate(self.layers):
#             a = l(next_layer)
#             next_layer = F.relu(a) if t<len(self.layers)-1 else a
#         return next_layer
#
#     def parameters(self):
#         return self.prs

#######################################################################
#########                     MODEL 3:                        #########
#######################################################################
# model = torch.nn.Sequential(
#         torch.nn.Linear(1, 200),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(200, 100),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(100, 1),
#     )

model = ANN([1, 30, 30, 30, 1])
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

batch_size = 128
epochs = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True, num_workers=2,)

pbar = tqdm(range(epochs))
for epoch in pbar:
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = model(batch_x)

        loss = loss_func(prediction, batch_y)
        pbar.set_description("Loss: %.4f" % loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
