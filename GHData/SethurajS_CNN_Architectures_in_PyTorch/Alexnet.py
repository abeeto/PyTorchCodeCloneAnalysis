# Importing the requirements
import torch 
import torch.nn as nn

# Building Alexnet Architecture
class Alexnet(nn.Module):
    def __init__(self, in_channels=3, out_classes=1000):
        super(Alexnet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_classes)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Object
    model = Alexnet(in_channels=3, out_classes=1000).to(device=device)

    # Sample data
    x = torch.randn(4, 3, 227, 227).to(device=device)

    print(model(x).shape) # [4, 1000]

