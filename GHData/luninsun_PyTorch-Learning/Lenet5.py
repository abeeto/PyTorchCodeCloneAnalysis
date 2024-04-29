import torch
from torch import nn
import torch.nn.functional as F


class Lenet5(nn.Module):
    """
    for lenet5 test
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_uint = nn.Sequential(
            # [b, 3, 32, 32]  =>  [b, 16, 27, 27]
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            # [b, 16, 28, 28] =>  [b, 16, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # [b, 16, 14, 14] =>  [b, 32, 10, 10]
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            # [b, 32, 10, 10] =>  [b, 32, 5, 5]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.fc_uint = nn.Sequential(
            nn.Linear(32 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return
        """

        batchsz = x.shape[0]
        # [b, 3, 32, 32] => [b, 32, 5, 5]
        x = self.conv_uint(x)

        # [b, 32, 5, 5] => [b, 32*5*5]
        x = x.view(batchsz, -1)

        # [b, 32*5*5] => [b, 10]
        logits = self.fc_uint(x)

        return logits


def main():
    net = Lenet5()
    print(net)


if __name__ == "__main__":
    main()