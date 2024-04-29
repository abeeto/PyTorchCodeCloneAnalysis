import numpy
import torch

# input 
X = numpy.random.uniform(-10, 10, 70).reshape(1, 7, -1)

'''
in_channels (int) – Number of channels in the input image
out_channels (int) – Number of channels produced by the convolution
kernel_size (int or tuple) – Size of the convolving kernel
stride (int or tuple, optional) – Stride of the convolution. Default: 1
padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
'''

class Simple1DCNN(torch.nn.Module):
    #https: //pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
    #                padding_mode='zeros', device=None, dtype=None)
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=7, out_channels=20, kernel_size=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)


        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs

model = Simple1DCNN().double()
print(model(torch.tensor(X)).shape)
