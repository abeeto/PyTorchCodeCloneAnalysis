import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary as tsummary

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv1_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv2_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv3_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv4_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        #self.fc0 = nn.Linear(512, 4096)
        #self.fc1 = nn.Linear(4096, 4096)
        #self.fc2 = nn.Linear(4096, 1000)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv0_0(x))
        x = F.relu(self.conv0_1(x))
        x = nn.MaxPool2d(2,2)(x) 
        x = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1_1(x))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.conv2_0(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.conv3_0(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.conv4_0(x))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        b, c, h, w = x.size()
        x = nn.AvgPool2d(h)(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        
        return x


def main():
    model = VGG19()
    print(tsummary.summary(model, torch.zeros((1,3,128,128)),batch_size=1, show_input=False))

    model = model.to(device=device)

    print('finish')



if __name__=='__main__':
    main()