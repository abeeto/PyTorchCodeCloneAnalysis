import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.nn.utils import weight_norm as norm

import numpy as np
from params import param

class Embed(nn.Module):
    def __init__(self, vocabSize, numUnits):
        # self.embedLayer = torch.nn.Linear(vocabSize, numUnits, bias=False)
        # nn.init.xavier_uniform(self.W.weight)
        '''
        num_embeddings : size of the dictionary of embeddings
        embedding_dim  : the size of each embedding vector
        '''
        super(Embed, self).__init__()
        self.embedLayer = nn.Embedding(num_embeddings = vocabSize,
                                       embedding_dim = numUnits,
                                       )


    def forward(self, input):
        embedOut = self.embedLayer(input)

        return embedOut

class Cv(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation, activationF = None, weightNorm = False):
        #nn.conv1d(in_channels, out_channels, kernel size, stride, padding,
        #          dilation, ...)
        #
        super(Cv, self).__init__()
        padDic = {"same" : (kernelSize-1)*dilation // 2, 
                  "causal" : (kernelSize-1)*dilation,
                  "none" : 0}
        self.pad = padding.lower()
        self.padValue = padDic[self.pad]
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.convOne = nn.Conv1d(in_channels=inChannel, 
                                 out_channels=outChannel, 
                                 kernel_size=kernelSize,
                                 stride=1,
                                 padding=self.padValue,
                                 dilation=dilation)
        if weightNorm:
            self.convOne = norm(self.convOne)

        self.activationF = activationF

        self.clear_buffer()

    def incremental_forward(self, input):
        # do convolutional process incrementally.
        # input: (B, C, T)
        # output : (B, C, input_buffer length)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        kw = self.convOne.kernel_size[0]
        dilation = self.convOne.dilation[0]

        bsz = input.size(0)  # input: bsz x dim x len
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, input.size(1), kw + (kw - 1) * (dilation - 1)) # if d = 1, third term will be 'kw' -> (bsz, dim, kw)
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :, :-1] = self.input_buffer[:, :, 1:].clone()
            # append next input
            self.input_buffer[:, :, -1] = input[:, :, -1]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, :, 0::dilation].contiguous()
        if str(input.device) == "cpu" and str(self.convOne.weight.device) == "cpu":
            if self.convOne.weight.requires_grad is True:
                self.convOne.weight.requires_grad = False
            output = torch.tensor(np.einsum("ijk,ljk->il", input, self.convOne.weight))
        else:
            output = torch.einsum('ijk,ljk->il', input, self.convOne.weight)

        output = output + self.convOne.bias

        output = torch.unsqueeze(output, dim=2)

        return output

    def clear_buffer(self):
        self.input_buffer = None

    def forward(self, input, is_incremental = False):
        if self.training or not is_incremental:
            cvOut = self.convOne(input)
            # In Causal mode, drop the right side of the outputs
            if self.pad == "causal" and self.padValue > 0:
                cvOut = cvOut[:, :, :-self.padValue]

            # activation Function
            if self.activationF in param.actFDic.keys():
                cvOut = param.actFDic[self.activationF](cvOut)
            elif self.activationF == None:
                pass
            else:
                raise ValueError("You should use appropriate actvation Function argument. \
                                [None, 'ReLU', 'sigmoid'].")
        
        else:
            cvOut = self.incremental_forward(input)
            # In Causal mode, drop the right side of the outputs
            if self.pad == "causal" and self.padValue > 0 and not is_incremental:
                cvOut = cvOut[:, :, :-self.padValue]
            # activation Function
            if self.activationF in param.actFDic.keys():
                cvOut = param.actFDic[self.activationF](cvOut)
            elif self.activationF == None:
                pass
            else:
                raise ValueError("You should use appropriate actvation Function argument. \
                                [None, 'ReLU', 'sigmoid'].")
        return cvOut

class Dc(nn.Module):
    '''
    Transposed Convolution 1d
    Lout = (Lin - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding*1
    '''
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation, activationF = None, weightNorm = False):
        super(Dc, self).__init__()
        padDic = {"same" : dilation*(kernelSize-1)//2,
                  "causal" : dilation*(kernelSize-1),
                  "none" : 0}
        self.pad = padding.lower()
        self.padValue = padDic[self.pad]
        self.transposedCv = nn.ConvTranspose1d(in_channels=inChannel,
                                               out_channels=outChannel,
                                               kernel_size=kernelSize,
                                               stride=2,
                                               padding=self.padValue,
                                               dilation=dilation)
        if weightNorm:
            self.convOne = norm(self.convOne)
        self.activationF = activationF

    def forward(self, input):
        DcOut = self.transposedCv(input)

        if self.pad == "causal":
            DcOut = DcOut[:, :, :-self.padValue]

        # activation Function
        if self.activationF in param.actFDic.keys():
            DcOut = param.actFDic[self.activationF](DcOut)
        elif self.activationF == None:
            pass
        else:
            raise ValueError("You should use appropriate actvation Function argument. \
                             [None, 'ReLU', 'sigmoid'].")
            
        return DcOut

class Hc(Cv):
    '''
    Highway Network and Convolution

    '''
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation, weightNorm=False):
        super(Hc, self).__init__(inChannel, outChannel*2, kernelSize,
                            padding, dilation, None, False)
    
    def forward(self, input, is_incremental=False):
        L = super(Hc, self).forward(input, is_incremental)
        H1, H2 = torch.chunk(L, 2, 1) # Divide L along axis 1 to get 2 matrices.
        self.Output = torch.sigmoid(H1) * H2 + (1-torch.sigmoid(H1)) * input

        return self.Output

    def clear_buffer(self):
        super(Hc, self).clear_buffer()

