import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer1.weight.data.fill_(1)
        self.layer1.bias.data.fill_(1)

        self.layer2 = nn.Linear(1, 1)
        self.layer2.weight.data.fill_(1)
        self.layer2.bias.data.fill_(1)

    def forward(self, x):
        h =  self.layer1(x)
        out = self.layer2(h)
        return out

def print_layers1(model):
    '''
    print out model layers
    :param model:
    :param numb_layers:
    :return:
    '''
    for index, module in enumerate(model.modules()):
        print("index:" + str(index) + " module:" + str(module.parameters))
        # for name2, params in child.named_parameters():
        #     print("name2:" + name2 + " params:" + str(params))

if __name__=="__main__":
    net = Net()
    print(net)
    # optimizer = optim.Adam([
    #             {'params': net.layer1.weight},
    #             {'params': net.layer1.bias, 'lr': 0.01},
    #             {'params': net.layer2.weight},
    #             {'params': net.layer2.bias, 'lr': 0.001}
    #                 ], lr=0.1, weight_decay=0.0001)
    optimizer = optim.Adam([
        {'params': net.layer1.weight, 'lr': 0.01},
        {'params': net.layer1.bias, 'lr': 0.01},
        {'params': net.layer2.weight, 'lr': 0.001},
        {'params': net.layer2.bias, 'lr': 0.001}
    ])
    out = net(Variable(torch.Tensor([[1]])))
    out.backward()
    optimizer.step()

    # print("weight", net.layer1.weight.data.numpy(), "grad", net.layer1.weight.grad.data.numpy())
    # print("bias", net.layer1.bias.data.numpy(), "grad", net.layer1.bias.grad.data.numpy())
    # print("weight", net.layer2.weight.data.numpy(), "grad", net.layer2.weight.grad.data.numpy())
    # print("bias", net.layer2.bias.data.numpy(), "grad", net.layer2.bias.grad.data.numpy())

    print("weight", net.layer1.weight.data.numpy(), "grad", net.layer1.weight.grad.data.numpy())
    print("bias", net.layer1.bias.data.numpy(), "grad", net.layer1.bias.grad.data.numpy())
    print("weight", net.layer2.weight.data.numpy(), "grad", net.layer2.weight.grad.data.numpy())
    print("bias", net.layer2.bias.data.numpy(), "grad", net.layer2.bias.grad.data.numpy())

    print(net.modules())
    # print_layers1(net)