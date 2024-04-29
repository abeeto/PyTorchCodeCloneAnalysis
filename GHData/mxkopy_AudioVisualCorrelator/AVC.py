import torch
import torchvision
import os

# Audio Visual Correlator Network


# We want to loss a sound frame against an image frame. 

# First we project the sound and image into a space where they share the same dimension. 
# This will be done through an autoencoder. 

# The network, given the sound, creates a test output in that space. Training is done
# over the MSE loss of test vectors and truth vectors. 

INIT_OUT_CHANNELS = 3

VID_INPUT_PARAMS  = (3, INIT_OUT_CHANNELS, 3, 1)
VID_OUTPUT_PARAMS = (INIT_OUT_CHANNELS, 3, 3, 1)

AUD_INPUT_PARAMS  = (2, INIT_OUT_CHANNELS, 3, 1)
AUD_OUTPUT_PARAMS = (INIT_OUT_CHANNELS, 2, 3, 1)

CHANNELS_ENC = [

    [INIT_OUT_CHANNELS, 8],
    [8, 12],
    [8, 8],
    [8, 12],
    [12, 16],
    [16, 16]

]

KERNELS = [

    [3, 2],
    [3, 1],
    [1, 1],
    [3, 1],
    [3, 1],
    [3, 1]

]

CHANNELS_DEC = [[pair[1], pair[0]] for pair in CHANNELS_ENC]

LAYERS_ENC = [channels + kernels for channels, kernels in zip(CHANNELS_ENC, KERNELS)]
LAYERS_DEC = [channels + kernels for channels, kernels in zip(CHANNELS_DEC, KERNELS)]


# determines the size of the 'bottleneck' between the encoder and decoder
BANDWIDTH_LIMIT = 16

CUTOFF_CHANNEL = CHANNELS_ENC[0][1]


class ImageEncoder(torch.nn.Module):


    def __init__(self, device=torch.device('cpu')):

        super(ImageEncoder, self).__init__()

        self.relu1 = torch.nn.ReLU().to(device)
        self.relu2 = torch.nn.ReLU().to(device)

        self.pool1 = torch.nn.AdaptiveAvgPool2d((512, 512)).to(device)
        self.pool2 = torch.nn.AdaptiveAvgPool2d((BANDWIDTH_LIMIT, BANDWIDTH_LIMIT)).to(device)
        
        self.dense = torch.nn.Linear(BANDWIDTH_LIMIT * BANDWIDTH_LIMIT, BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        self.conv1 = torch.nn.Conv2d(*VID_INPUT_PARAMS).to(device)
        self.conv2 = torch.nn.Conv2d(*LAYERS_ENC[0]).to(device)
        self.conv3 = torch.nn.Conv2d(*LAYERS_ENC[1]).to(device)



    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        # out = self.relu1(out)
        # out = self.conv3(out)

        out = self.pool2(out).view(-1, CUTOFF_CHANNEL, BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)
        out = self.dense(out)

        return out



class ImageDecoder(torch.nn.Module):


    def __init__(self, image_size, device=torch.device('cpu')):

        super(ImageDecoder, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool2d(image_size).to(device)

        self.deconv5 = torch.nn.ConvTranspose2d(*LAYERS_DEC[1]).to(device)
        self.deconv6 = torch.nn.ConvTranspose2d(*LAYERS_DEC[0]).to(device)
        self.deconv7 = torch.nn.ConvTranspose2d(*VID_OUTPUT_PARAMS).to(device)

    def forward(self, x):

        out = x.view(-1, CUTOFF_CHANNEL, BANDWIDTH_LIMIT, BANDWIDTH_LIMIT)

        # out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)

        out = self.pool(out)

        return out



class AudioEncoder(torch.nn.Module):


    def __init__(self, device=torch.device('cpu')):
        
        super(AudioEncoder, self).__init__()
        
        self.pool1 = torch.nn.AdaptiveAvgPool1d(10000)
        self.pool2 = torch.nn.AdaptiveAvgPool1d(BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        self.relu1 = torch.nn.ReLU().to(device)

        self.conv1 = torch.nn.Conv1d(*AUD_INPUT_PARAMS).to(device)
        self.conv2 = torch.nn.Conv1d(*LAYERS_ENC[0]).to(device)
        # self.conv3 = torch.nn.Conv1d(*LAYERS_ENC[1]).to(device)

    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        # out = self.relu1(out)
        # out = self.conv3(out)

        out = self.pool2(out)

        return out



class AudioDecoder(torch.nn.Module):

    
    def __init__(self, frame_size, device=torch.device('cpu')):

        super(AudioDecoder, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool1d(frame_size)

        self.deconv5 = torch.nn.ConvTranspose1d(*LAYERS_DEC[1]).to(device)
        self.deconv6 = torch.nn.ConvTranspose1d(*LAYERS_DEC[0]).to(device)
        self.deconv7 = torch.nn.ConvTranspose1d(*AUD_OUTPUT_PARAMS).to(device)

    def forward(self, x):

        out = x.view(-1, CUTOFF_CHANNEL, BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        self.dense = torch.nn.Linear(BANDWIDTH_LIMIT * BANDWIDTH_LIMIT, BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        # out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)

        out = self.pool(out)

        return out 


