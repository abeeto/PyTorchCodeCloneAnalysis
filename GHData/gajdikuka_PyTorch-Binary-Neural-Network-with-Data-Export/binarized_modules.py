import torch
import torch.nn as nn

def Binarize(tensor,quant_mode='det'):
        tensor = torch.where(tensor==0,torch.tensor(1.).cuda(),tensor)
        tensor = tensor.sign()
        return tensor
    
class BinarizeSign(nn.Module):

    def __init__(self):
        super(BinarizeSign, self).__init__()

    def forward(self, input):

        input.data=Binarize(input.data)
        out = input
        return out

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out