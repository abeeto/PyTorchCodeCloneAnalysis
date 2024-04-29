import torch
import torch.nn as nn
import torch.nn.functional as F

class Correlation(nn.Module):
    def __init__(self, kernel_size=1,max_displacement=1,stride1=1,stride2=1, pad=1,*args, **kwargs):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = (2*max_displacement)//stride2+1
        self.pad_size = self.max_displacement
        self.stride1 = stride1
        self.stride2 = stride2

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.size()
        x1=x1[:,:,0:H1:self.stride2,0:W1:self.stride2]
        x2 = F.pad(x2, [self.pad_size] * 4)
        B2, C2, H2, W2 = x2.size()
        x2=x2[:,:,0:H2:self.stride2,0:W2:self.stride2]
        cv = []
        for i in range(0,self.output_dim):
            for j in range(0,self.output_dim):
                cost = x1 * x2[:, :, i:(i + H1//self.stride2), j:(j + W1//self.stride2)]
                print(cost.shape)
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)