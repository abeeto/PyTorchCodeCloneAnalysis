import torch.nn as nn


class VGG_Net(nn.Module):
    def __init__(self,
                 num_classes: int,
                 net_type: str = "VGG-11",
                 in_channels: int = 3,
                 input_width: int = 224,
                 ):

        super(VGG_Net, self).__init__()

        self.VGG = dict()
        self.VGG["VGG-11"] = [1, 1, 2, 2, 2]
        self.VGG["VGG-13"] = [2, 2, 2, 2, 2]
        self.VGG["VGG-16"] = [2, 2, 3, 3, 3]
        self.VGG["VGG-19"] = [2, 2, 4, 4, 4]

        # Adding convolutional layers
        layers = []
        out_channels = 64
        for layer_numbers in self.VGG[net_type]:
            for _ in range(layer_numbers):
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=1,
                              padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()]
                in_channels = out_channels

            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
            if (out_channels != 512):
                out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)

        # Adding fc layers
        in_fc: int = input_width//32
        self.fc_layers = nn.Sequential(
            nn.Linear(in_fc*in_fc*512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x
