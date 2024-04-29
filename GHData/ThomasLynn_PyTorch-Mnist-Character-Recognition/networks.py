import torch
        
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #32x32
        sizes = [1,200,400,600,800,1000]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #16x16
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #8x8
            torch.nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #4x4
            torch.nn.Conv2d(sizes[3], sizes[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #2x2
            torch.nn.Conv2d(sizes[4], sizes[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(sizes[5], sizes[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #1x1
            torch.nn.Dropout2d(),
            torch.nn.Flatten(),
            torch.nn.Linear(sizes[5], 10))
        
    def forward(self, x):
        out = self.layers(x)
        return out
