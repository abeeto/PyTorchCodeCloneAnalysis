import torch.nn


class FFNet(torch.nn.Module):
    def __init__(self, d_in, num_hidden, d_out):
        super(FFNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden, d_out)

    def forward(self, x):
        h = torch.nn.functional.relu(self.linear1(x))
        return self.linear2(h)

