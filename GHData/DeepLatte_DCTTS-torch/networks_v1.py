import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms


import numpy as np
from params import param
import module

# torch.nn.Sequential

class TextEnc(nn.Module):
    def __init__(self, vocabSize, embSize, dSize):
        super(TextEnc, self).__init__()

        self.EmbLayer = module.Embed(vocabSize, embSize) # (1 , T) -> (V, T), (e, V) * (V, T) -> (e, T)
        self.seq = nn.ModuleList()
        self.Conv1st = module.Cv(inChannel = embSize,
                                 outChannel = dSize * 2,
                                 kernelSize = 1,
                                 padding = "SAME",
                                 dilation = 1,
                                 activationF = "ReLU")
        self.seq.append(self.Conv1st)
        self.seq.append(nn.Dropout(param.dr))
        
        self.Conv2nd = module.Cv(inChannel = dSize * 2,
                                 outChannel = dSize * 2,
                                 kernelSize = 1,
                                 padding = "SAME",
                                 dilation = 1,
                                 activationF = None)
        self.seq.append(self.Conv2nd)
        self.seq.append(nn.Dropout(param.dr))
        
        # HcTwice
        for _ in range(2):
            for d in range(4):
                self.seq.append(module.Hc(inChannel = dSize*2,
                                        outChannel = dSize*2,
                                        kernelSize = 3,
                                        padding = "SAME",
                                        dilation = 3 ** d))
                
                self.seq.append(nn.Dropout(param.dr))

        for _ in range(2):
            self.seq.append(module.Hc(inChannel = dSize*2,
                                      outChannel = dSize*2,
                                      kernelSize = 3,
                                      padding = "SAME",
                                      dilation = 1))
            
            self.seq.append(nn.Dropout(param.dr))
        
        for _ in range(2):
            self.seq.append(module.Hc(inChannel = dSize*2,
                                      outChannel = dSize*2,
                                      kernelSize = 1,
                                      padding = "SAME",
                                      dilation = 1))
            
            # self.seq.append(nn.Dropout(param.dr))

    def forward(self, input):
        x = self.EmbLayer(input)
        x = x.transpose(1,2) # (B, e, T)
        for f in self.seq:
            x = f(x)

        K, V = torch.chunk(x, 2, 1) # Divide txtEncOut along axis 1 to get 2 matrices. 

        return K, V

class AudioEnc(nn.Module):
    def __init__(self, fbinSize, dSize):
        super(AudioEnc, self).__init__()
        self.seq = nn.ModuleList()
        self.Conv1st = module.Cv(inChannel = fbinSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = "ReLU")
        self.seq.append(self.Conv1st)
        
        self.seq.append(nn.Dropout(param.dr))

        self.Conv2nd = module.Cv(inChannel = dSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = "ReLU")
        self.seq.append(self.Conv2nd)
        
        self.seq.append(nn.Dropout(param.dr))

        self.Conv3rd = module.Cv(inChannel = dSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = None)
        self.seq.append(self.Conv3rd)
        
        self.seq.append(nn.Dropout(param.dr))
        
        for _ in range(2):
            for d in range(4):
                self.seq.append(module.Hc(inChannel = dSize,
                                          outChannel = dSize,
                                          kernelSize = 3,
                                          padding = "causal",
                                          dilation = 3 ** d))
                
                self.seq.append(nn.Dropout(param.dr))

        for _ in range(2):
            self.seq.append(module.Hc(inChannel = dSize,
                                      outChannel = dSize,
                                      kernelSize = 3,
                                      padding = "causal",
                                      dilation = 3))


    def forward(self, input, is_incremental):
        x = input.transpose(1,2) # (B, n_mels, T)
        for f in self.seq:
            if isinstance(f, module.Cv) or isinstance(f, module.Hc):
                x = f(x, is_incremental)
            else:
                x = f(x)
        return x
    
    def clear_buffer(self):
        for module in self.seq._modules.values():
            try:
                module.clear_buffer()
            except:
                pass

class AudioDec(nn.Module):
    def __init__(self, fbinSize, dSize):
        super(AudioDec, self).__init__()
        self.seq = nn.ModuleList()
        self.Conv1st = module.Cv(inChannel = dSize * 2,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = None)
        self.seq.append(self.Conv1st)
        
        self.seq.append(nn.Dropout(param.dr))

        for d in range(4):
            self.seq.append(module.Hc(inChannel = dSize,
                                    outChannel = dSize,
                                    kernelSize = 3,
                                    padding = "causal",
                                    dilation = 3 ** d))
            
            self.seq.append(nn.Dropout(param.dr))

        for _ in range(2):
            self.seq.append(module.Hc(inChannel = dSize,
                                      outChannel = dSize,
                                      kernelSize = 3,
                                      padding = "causal",
                                      dilation = 1))
            
            self.seq.append(nn.Dropout(param.dr))

        for _ in range(3):
            self.seq.append(module.Cv(inChannel = dSize,
                                      outChannel = dSize,
                                      kernelSize = 1,
                                      padding = "causal",
                                      dilation = 1,
                                      activationF = "ReLU"))
            
            self.seq.append(nn.Dropout(param.dr))

        self.ConvLast = module.Cv(inChannel = dSize,
                                  outChannel = fbinSize,
                                  kernelSize = 1,
                                  padding = "causal",
                                  dilation = 1,
                                  activationF = "sigmoid")
        self.seq.append(self.ConvLast)

    def forward(self, input, is_incremental=False):
        x = input
        for f in self.seq:
            if isinstance(f, module.Cv) or isinstance(f, module.Hc):
                x = f(x, is_incremental)
            else:
                x = f(x)

        return x
        
    def clear_buffer(self):
        for module in self.seq._modules.values():
            try:
                module.clear_buffer()
            except:
                pass

class SSRN(nn.Module):
    def __init__(self, upsamfbinSize, fbinSize, c,  dSize):
        super(SSRN, self).__init__()
        self.seq = nn.ModuleList()
        self.Conv1st = module.Cv(inChannel = fbinSize,
                                  outChannel = c,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None)
        self.seq.append(self.Conv1st)
        
        self.seq.append(nn.Dropout(param.dr))
        
        for d in range(2):
            self.seq.append(module.Hc(inChannel = c,
                                      outChannel = c,
                                      kernelSize = 3,
                                      padding = "SAME",
                                      dilation = 3 ** d))
            
            self.seq.append(nn.Dropout(param.dr))

        for _ in range(2):
            self.seq.append(module.Dc(inChannel = c,
                                      outChannel = c,
                                      kernelSize = 2,
                                      padding = "SAME",
                                      dilation = 1,
                                      activationF = None))
            
            self.seq.append(nn.Dropout(param.dr))
            
            for _ in range(2):
                self.seq.append(module.Hc(inChannel = c,
                                          outChannel = c,
                                          kernelSize = 3,
                                          padding = "SAME",
                                          dilation = 1))
                
                self.seq.append(nn.Dropout(param.dr))

        self.seq.append(module.Cv(inChannel = c,
                                  outChannel = 2 * c,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None))
        
        self.seq.append(nn.Dropout(param.dr))
        
        for _ in range(2):
            self.seq.append(module.Hc(inChannel = 2*c,
                                      outChannel = 2*c,
                                      kernelSize = 3,
                                      padding = "SAME",
                                      dilation = 1))
            
            self.seq.append(nn.Dropout(param.dr))

        self.seq.append(module.Cv(inChannel = 2 * c,
                                  outChannel = upsamfbinSize,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None))
        self.seq.append(nn.Dropout(param.dr))

        for _ in range(2):
            self.seq.append(module.Cv(inChannel = upsamfbinSize,
                                      outChannel = upsamfbinSize,
                                      kernelSize = 1,
                                      padding = "SAME",
                                      dilation = 1,
                                      activationF = "ReLU"))
            
            self.seq.append(nn.Dropout(param.dr))

        self.ConvLast = module.Cv(inChannel = upsamfbinSize,
                                outChannel = upsamfbinSize,
                                kernelSize = 1,
                                padding = "SAME",
                                dilation = 1,
                                activationF = "sigmoid")
        self.seq.append(self.ConvLast)

    def forward(self, input, is_incremental=False):
        x = input.transpose(1,2) # (B, n_mels, T/r)
        for f in self.seq:
            if isinstance(f, module.Cv):
                x = f(x, is_incremental)
            else:
                x = f(x)

        return x # (B, n_mag, T)

class AttentionNet(nn.Module):
    '''
    input:
        K : Keys (B, d, N)
        V : Valuse (B, d, N)
        Q : Queries (B, d, T/r)
    return:
        R_ : R' (B, 2*d, T/r)
        A : Attention matrix (N, T/r)
    '''
    def __init__(self):
        super(AttentionNet, self).__init__()
    
    def forward(self, K, V, Q):
        A = torch.softmax((torch.bmm(K.transpose(1,2), Q) / np.sqrt(param.d)), dim=1) 
        # (B, N, T/r), Alignment, softmax along axis 1
        R = torch.bmm(V, A) # (B, d, T/r)
        R_ = torch.cat((R, Q), dim=1) # (B, 2*d, T/r)

        maxAtt = torch.argmax(A, 1)

        return R_, A, maxAtt

class t2mGraph(nn.Module):
    def __init__(self):
        super(t2mGraph, self).__init__()
        self.TextEnc = TextEnc(param.vocabSize, param.e, param.d)
        self.AudioEnc = AudioEnc(param.n_mels, param.d)
        self.AudioDec = AudioDec(param.n_mels, param.d)
        self.AttentionNet = AttentionNet()

    def forward(self, textInput, melInput, is_incremental=False):
        K, V = self.TextEnc(textInput) # K, V: (B, d, N)
        Q = self.AudioEnc(melInput, is_incremental) # Q : (B, d, T/r)
        R_, Alignment, maxAtt = self.AttentionNet(K, V, Q) # R_ : (B, 2*d, T/r)

        coarseMel = self.AudioDec(R_, is_incremental) # coarseMel : (B, n_mels, T/r)

        return coarseMel, Alignment, maxAtt

class SSRNGraph(nn.Module):
    def __init__(self):
        super(SSRNGraph, self).__init__()
        self.SSRN = SSRN(param.n_mags, param.n_mels, param.c, param.d)

    def forward(self, input, is_incremental=False):
        SSRNOut = self.SSRN(input, is_incremental)

        return SSRNOut