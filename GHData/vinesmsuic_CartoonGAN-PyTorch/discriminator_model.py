import torch
import torch.nn as nn

# PyTorch implementation by vinesmsuic
# The paper claimed to use BatchNorm and Leaky ReLu.
# But here we use InstanceNorm instead of BatchNorm.
# We also use reflect padding instead of constant padding here (to reduce artifacts?)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.model = nn.Sequential(
            #k3n32s1
            nn.Conv2d(in_channels,features[0],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s2
            nn.Conv2d(features[0],features[1],kernel_size=3,stride=2,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1
            nn.Conv2d(features[1],features[2],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s2
            nn.Conv2d(features[2],features[2],kernel_size=3,stride=2,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n256s1
            nn.Conv2d(features[2],features[3],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n256s1
            nn.Conv2d(features[3],features[3],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n1s1
            nn.Conv2d(features[3],out_channels,kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
        )

    def forward(self, x):
        x = self.model(x)
        return x
        #No sigmoid for LSGAN adv loss
        #return torch.sigmoid(x)



def test():
    x = torch.randn((5, 3, 256 ,256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()





