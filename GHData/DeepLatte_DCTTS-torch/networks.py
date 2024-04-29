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
        self.EmbLayer = module.Embed(vocabSize, embSize)

        self.Conv1st = module.Cv(inChannel = embSize,
                                 outChannel = dSize * 2,
                                 kernelSize = 1,
                                 padding = "SAME",
                                 dilation = 1,
                                 activationF = "ReLU")
        
        self.Conv2nd = module.Cv(inChannel = dSize * 2,
                                 outChannel = dSize * 2,
                                 kernelSize = 1,
                                 padding = "SAME",
                                 dilation = 1,
                                 activationF = None)

        self.HcTwice1 = nn.Sequential(module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 1),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 3),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 9),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 27))
        self.HcTwice2 = nn.Sequential(module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 1),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 3),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 9),
                                      module.Hc(inChannel = dSize*2,
                                                outChannel = dSize*2,
                                                kernelSize = 3,
                                                padding = "SAME",
                                                dilation = 27))
        self.Hc3 = nn.Sequential(module.Hc(inChannel = dSize*2,
                                           outChannel = dSize*2,
                                           kernelSize = 3,
                                           padding = "SAME",
                                           dilation = 1),
                                 module.Hc(inChannel = dSize*2,
                                           outChannel = dSize*2,
                                           kernelSize = 3,
                                           padding = "SAME",
                                           dilation = 1))
        self.Hc4 = nn.Sequential(module.Hc(inChannel = dSize*2,
                                           outChannel = dSize*2,
                                           kernelSize = 1,
                                           padding = "SAME",
                                           dilation = 1),
                                 module.Hc(inChannel = dSize*2,
                                           outChannel = dSize*2,
                                           kernelSize = 1,
                                           padding = "SAME",
                                           dilation = 1))

    def forward(self, input):
        txtEncOut = self.EmbLayer(input)
        txtEncOut = txtEncOut.transpose(1,2) # (B, e, T)
        txtEncOut = self.Conv1st(txtEncOut)
        txtEncOut = self.Conv2nd(txtEncOut)
        txtEncOut = self.HcTwice1(txtEncOut)
        txtEncOut = self.HcTwice2(txtEncOut)
        txtEncOut = self.Hc3(txtEncOut)
        txtEncOut = self.Hc4(txtEncOut)
        K, V = torch.chunk(txtEncOut, 2, 1) # Divide txtEncOut along axis 1 to get 2 matrices. 

        return K, V

class AudioEnc(nn.Module):
    def __init__(self, fbinSize, dSize):
        super(AudioEnc, self).__init__()
        self.Conv1st = module.Cv(inChannel = fbinSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = "ReLU")
        self.Conv2nd = module.Cv(inChannel = dSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = "ReLU")
        self.Conv3rd = module.Cv(inChannel = dSize,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = None)
        self.HcTwice1 = nn.Sequential(module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 1),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 3),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 9),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 27))
        self.HcTwice2 = nn.Sequential(module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 1),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 3),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 9),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 27))
        self.Hc3 = nn.Sequential(module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 3),
                                module.Hc(inChannel = dSize,
                                        outChannel = dSize,
                                        kernelSize = 3,
                                        padding = "causal",
                                        dilation = 3))

    def forward(self, input):
        input = input.transpose(1,2) # (B, n_mels, T)
        audEncOut = self.Conv1st(input)
        audEncOut = self.Conv2nd(audEncOut)
        audEncOut = self.Conv3rd(audEncOut)
        audEncOut = self.HcTwice1(audEncOut)
        audEncOut = self.HcTwice2(audEncOut)
        audEncOut = self.Hc3(audEncOut)

        return audEncOut
            
class AudioDec(nn.Module):
    def __init__(self, fbinSize, dSize):
        super(AudioDec, self).__init__()
        self.Conv1st = module.Cv(inChannel = dSize * 2,
                                 outChannel = dSize,
                                 kernelSize = 1,
                                 padding = "causal",
                                 dilation = 1,
                                 activationF = None)
        self.Hc1 = nn.Sequential(module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 1),
                                 module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 3),
                                 module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 9),
                                 module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 27))
        self.Hc2 = nn.Sequential(module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 1),
                                 module.Hc(inChannel = dSize,
                                            outChannel = dSize,
                                            kernelSize = 3,
                                            padding = "causal",
                                            dilation = 1))
        self.ConvThrice = nn.Sequential(module.Cv(inChannel = dSize,
                                                  outChannel = dSize,
                                                  kernelSize = 1,
                                                  padding = "causal",
                                                  dilation = 1,
                                                  activationF = "ReLU"),
                                        module.Cv(inChannel = dSize,
                                                  outChannel = dSize,
                                                  kernelSize = 1,
                                                  padding = "causal",
                                                  dilation = 1,
                                                  activationF = "ReLU"),
                                        module.Cv(inChannel = dSize,
                                                  outChannel = dSize,
                                                  kernelSize = 1,
                                                  padding = "causal",
                                                  dilation = 1,
                                                  activationF = "ReLU"))
        self.ConvLast = module.Cv(inChannel = dSize,
                                  outChannel = fbinSize,
                                  kernelSize = 1,
                                  padding = "causal",
                                  dilation = 1,
                                  activationF = "sigmoid")
    def forward(self, input):
        decOutput = self.Conv1st(input)
        decOutput = self.Hc1(decOutput)
        decOutput = self.Hc2(decOutput)
        decOutput = self.ConvThrice(decOutput)
        decOutput = self.ConvLast(decOutput)

        return decOutput
        

class SSRN(nn.Module):
    def __init__(self, upsamfbinSize, fbinSize, c,  dSize):
        super(SSRN, self).__init__()
        self.Conv1st = module.Cv(inChannel = fbinSize,
                                  outChannel = c,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None)
        self.Hc1 = nn.Sequential(module.Hc(inChannel = c,
                                            outChannel = c,
                                            kernelSize = 3,
                                            padding = "SAME",
                                            dilation = 1),
                                 module.Hc(inChannel = c,
                                            outChannel = c,
                                            kernelSize = 3,
                                            padding = "SAME",
                                            dilation = 3))
        self.DcHcTwice1 = nn.Sequential(module.Dc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 2,
                                                  padding = "SAME",
                                                  dilation = 1,
                                                  activationF = None),
                                        module.Hc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 3,
                                                  padding = "SAME",
                                                  dilation = 1),
                                        module.Hc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 3,
                                                  padding = "SAME",
                                                  dilation = 3))
        self.DcHcTwice2 = nn.Sequential(module.Dc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 2,
                                                  padding = "SAME",
                                                  dilation = 1,
                                                  activationF = None),
                                        module.Hc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 3,
                                                  padding = "SAME",
                                                  dilation = 1),
                                        module.Hc(inChannel = c,
                                                  outChannel = c,
                                                  kernelSize = 3,
                                                  padding = "SAME",
                                                  dilation = 3))
        self.Conv2nd = module.Cv(inChannel = c,
                                  outChannel = 2 * c,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None)
        self.HcTwice = nn.Sequential(module.Hc(inChannel = 2*c,
                                               outChannel = 2*c,
                                               kernelSize = 3,
                                               padding = "SAME",
                                               dilation = 1),
                                     module.Hc(inChannel = 2*c,
                                               outChannel = 2*c,
                                               kernelSize = 3,
                                               padding = "SAME",
                                               dilation = 1))
        self.Conv3rd = module.Cv(inChannel = 2 * c,
                                  outChannel = upsamfbinSize,
                                  kernelSize = 1,
                                  padding = "SAME",
                                  dilation = 1,
                                  activationF = None)
        self.ConvTwice = nn.Sequential(module.Cv(inChannel = upsamfbinSize,
                                                  outChannel = upsamfbinSize,
                                                  kernelSize = 1,
                                                  padding = "SAME",
                                                  dilation = 1,
                                                  activationF = "ReLU"),
                                        module.Cv(inChannel = upsamfbinSize,
                                                  outChannel = upsamfbinSize,
                                                  kernelSize = 1,
                                                  padding = "SAME",
                                                  dilation = 1,
                                                  activationF = "ReLU"))
        self.ConvLast = module.Cv(inChannel = upsamfbinSize,
                                outChannel = upsamfbinSize,
                                kernelSize = 1,
                                padding = "SAME",
                                dilation = 1,
                                activationF = "sigmoid")

    def forward(self, input):
        input = input.transpose(1,2) # (B, n_mels, T/r)
        ssrnOut = self.Conv1st(input)
        ssrnOut = self.Hc1(ssrnOut)
        ssrnOut = self.DcHcTwice1(ssrnOut)
        ssrnOut = self.DcHcTwice2(ssrnOut)
        ssrnOut = self.Conv2nd(ssrnOut)
        ssrnOut = self.HcTwice(ssrnOut)
        ssrnOut = self.Conv3rd(ssrnOut)
        ssrnOut = self.ConvTwice(ssrnOut)
        ssrnOut = self.ConvLast(ssrnOut)
        
        return ssrnOut # (B, n_mag, T)

class AttentionNet(nn.Module):
    '''
    input:
        K : Keys (B, d, N)
        V : Valuse (B, d, N)
        Q : Queries (B, d, T/r)
    return:
        R_ : R' (B, 2*d, T/r)
        A : Attention matrix (B, N, T/r)
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

    def forward(self, textInput, melInput):
        K, V = self.TextEnc(textInput) # K, V: (B, d, N)
        Q = self.AudioEnc(melInput) # Q : (B, d, T/r)
        R_, Alignment, maxAtt = self.AttentionNet(K, V, Q) # R_ : (B, 2*d, T/r)

        coarseMel = self.AudioDec(R_) # coarseMel : (B, n_mels, T/r)

        return coarseMel, Alignment, maxAtt

class SSRNGraph(nn.Module):
    def __init__(self):
        super(SSRNGraph, self).__init__()
        self.SSRN = SSRN(param.n_mags, param.n_mels, param.c, param.d)

    def forward(self, input):
        SSRNOut = self.SSRN(input)

        return SSRNOut