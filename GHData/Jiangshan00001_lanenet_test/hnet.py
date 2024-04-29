from torch import nn

HNET_DEFAULT_SIZE = (64, 128)
import torch
DEFAULT_SIZE = (256, 512)

class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        # TODO
        self.cbr0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU())
        self.cbr1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU())
        self.mp1=nn.MaxPool2d(3, stride=2, padding=1)
        self.cbr2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU())
        self.cbr3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU())
        self.mp2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.cbr4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU())
        self.cbr5 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU())
        self.mp3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.lbr1 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm2d(16),
            nn.PReLU())
        self.final_l=nn.Linear(1024,6)

    def forward(self, x):
      # TODO
      x_out = self.cbr0.forward(x)
      x_out = self.cbr1.forward(x_out)
      x_out = self.mp1.forward(x_out)
      x_out = self.cbr2.forward(x_out)
      x_out = self.cbr3.forward(x_out)

      x_out = self.mp2.forward(x_out)
      x_out = self.cbr4.forward(x_out)
      x_out = self.cbr5.forward(x_out)

      x_out = self.mp3.forward(x_out)
      x_out = self.lbr1.forward(x_out)
      x_out = self.final_l.forward(x_out)


class HomographyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, size=DEFAULT_SIZE):
        # TODO
        pass

    def __getitem__(self, idx):
        # TODO
        return image, ground_truth_trajectory

    def __len__(self):
        # TODO
        pass