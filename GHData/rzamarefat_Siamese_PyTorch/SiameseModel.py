import torch

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(1, 4, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(4),
            
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(4, 8, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(8),


            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(8),)

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(8*100*100, 500),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(500, 500),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(500, 5))

    def forward(self, input1, input2):
        output1 = self.cnn1(input1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc1(output1)


        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)
        output2 = self.fc1(output2)

        return output1, output2