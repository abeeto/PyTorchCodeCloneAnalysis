from torch import nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # CIFA10 model结构
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 如果想要验证一下我们的model写的对不对，可以模拟输入X，跑一边看看输出的shape
if __name__ == '__main__':
    model = MyModule()
    input = torch.randn(64,3,32,32)
    output = model(input)
    print(output.shape) # torch.Size([64, 10])