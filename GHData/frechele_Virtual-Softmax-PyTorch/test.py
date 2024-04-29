import torch
from virtsm.models.modules import VirtualLinear

if __name__ == '__main__':
    net = VirtualLinear(10, 5)
    net.train()

    batch_size = 3
    input = torch.randn(batch_size, 10)
    target = torch.zeros(batch_size, dtype=torch.long).random_(5)

    output = net(input, target)

    print(output)
    print(output.shape)

