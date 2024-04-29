import torch

class SineNet(torch.nn.Module):
    def __init__(self, n_hiden_neurons):
        super().__init__()
        NUMNE = n_hiden_neurons
        self.fc1 = torch.nn.Linear(1, NUMNE)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(NUMNE, NUMNE)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(NUMNE, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

sine_net = SineNet(int(input()))
sine_net.forward(torch.Tensor([1.]))

print(sine_net)
