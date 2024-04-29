import torch
import torch.nn as nn
import torchsnooper

class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()    
        self.L1 = nn.Conv3d(in_channels = 1,
                            out_channels = 4,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L2 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L3 = nn.Conv3d(in_channels = 8,
                            out_channels = 16,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L4 = nn.Conv3d(in_channels = 16,
                            out_channels = 32,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L5 = nn.Conv3d(in_channels = 32,
                            out_channels = 64,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)

        self.MP = nn.MaxPool3d(kernel_size = 2)

        self.FC11 = nn.Linear(in_features=64,out_features=32)
        self.FC21 = nn.Linear(in_features=64,out_features=32)
        self.FC31 = nn.Linear(in_features=64,out_features=32)
        self.FC41 = nn.Linear(in_features=64,out_features=32)
        self.FC51 = nn.Linear(in_features=64,out_features=32)
        self.FC61 = nn.Linear(in_features=64,out_features=32)

        self.FC12 = nn.Linear(in_features=32,out_features=16)
        self.FC22 = nn.Linear(in_features=32,out_features=16)
        self.FC32 = nn.Linear(in_features=32,out_features=16)
        self.FC42 = nn.Linear(in_features=32,out_features=16)
        self.FC52 = nn.Linear(in_features=32,out_features=16)
        self.FC62 = nn.Linear(in_features=32,out_features=16)

        self.FC13 = nn.Linear(in_features=16,out_features=4)
        self.FC23 = nn.Linear(in_features=16,out_features=4)
        self.FC33 = nn.Linear(in_features=16,out_features=4)
        self.FC43 = nn.Linear(in_features=16,out_features=4)
        self.FC53 = nn.Linear(in_features=16,out_features=4)
        self.FC63 = nn.Linear(in_features=16,out_features=4)

        self.AF = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    # @torchsnooper.snoop()
    def forward(self, input):
        out = self.AF(self.MP(self.L1(input)))
        out = self.AF(self.MP(self.L2(out)))
        out = self.AF(self.MP(self.L3(out)))
        out = self.AF(self.MP(self.L4(out)))
        out = self.AF(self.MP(self.L5(out)))
        out = out.view(-1,64)

        # o1,o2,o3 are for symmetry plane
        # o4,o5,o6 are for rotation axis
        o1 = self.AF(self.FC11(out))
        o2 = self.AF(self.FC21(out))
        o3 = self.AF(self.FC31(out))
        o4 = self.AF(self.FC41(out))
        o5 = self.AF(self.FC51(out))
        o6 = self.AF(self.FC61(out))

        o1 = self.AF(self.FC12(o1))
        o2 = self.AF(self.FC22(o2))
        o3 = self.AF(self.FC32(o3))
        o4 = self.AF(self.FC42(o4))
        o5 = self.AF(self.FC52(o5))
        o6 = self.AF(self.FC62(o6))

        o1 = self.AF(self.FC13(o1))
        o2 = self.AF(self.FC23(o2))
        o3 = self.AF(self.FC33(o3))
        o4 = self.AF(self.FC43(o4))
        o5 = self.AF(self.FC53(o5))
        o6 = self.AF(self.FC63(o6))

        o1 = o1/torch.norm(o1, dim = 0)
        o2 = o2/torch.norm(o2, dim = 0)
        o3 = o3/torch.norm(o3, dim = 0)
        o4 = o4/torch.norm(o4, dim = 0)
        o5 = o5/torch.norm(o5, dim = 0)
        o6 = o6/torch.norm(o6, dim = 0)

        self.batchoutput = torch.zeros(o1.shape[0], 6, 4).cuda()
        self.reshapeOutput(o1, 1)
        self.reshapeOutput(o2, 2)
        self.reshapeOutput(o3, 3)
        self.reshapeOutput(o4, 4)
        self.reshapeOutput(o5, 5)
        self.reshapeOutput(o6, 6)

        return self.batchoutput
        # reshape output
    def reshapeOutput(self, output, pos):
        for i in range(self.batchoutput.shape[0]):
            self.batchoutput[i][pos - 1] = output[i]

