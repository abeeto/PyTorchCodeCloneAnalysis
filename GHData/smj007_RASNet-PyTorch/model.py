import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Within the decoder, there are 1x1 Conv to "reduce complexity"
        # Paper does not mention how much the reduction is, taking as n = 4

        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.mid_channels = in_channels // 4

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, 1)
        self.norm1 = nn.BatchNorm2d(self.mid_channels)

        self.deconv2 = nn.ConvTranspose2d(self.mid_channels, self.mid_channels, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(self.mid_channels)

        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels, 1)
        self.norm3 = nn.BatchNorm2d(self.out_channels)


    # Optional: Add ReLU (Not mentioned in the paper)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.deconv2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        return x     


class AFM(nn.Module):
  def __init__(self, in_channels, low_channels): # low = in
    super().__init__()
    self.GP = nn.AdaptiveAvgPool2d(1)

    # High-level features
    self.conv_h = nn.Conv2d(in_channels, low_channels, 1, padding=0)
    self.norm = nn.BatchNorm2d(low_channels)
    self.softmax = nn.Softmax(dim=1)

    # Low-level features
    # Paper claims 3x3, but will result in dimension mismatch (For addition)
    # Setting to 1x1 conv

    # self.conv_l = nn.Conv2d(low_channels, low_channels, 3, padding=0)
    self.conv_l = nn.Conv2d(low_channels, low_channels, 1, padding=0)

  # Example - Fig: Up (3x3, //1 channels); In (1x1, //1 channels)
  # Low-level feature maps are weighted
  # High-level feature maps create weights, and unchanged version is added in the end 
  
  def forward(self, x_high, x_low):
    mid_high = self.GP(x_high)
    mid_high = self.norm(self.conv_h(mid_high))

    # Weights
    weights = self.softmax(mid_high)
    weighted_low = weights.mul(self.conv_l(x_low))

    # Add and return
    return x_high + weighted_low

class RASNet(nn.Module):
  def __init__(self, num_classes=8, num_channels=3, pretrained=True):
    super().__init__()

    # Pretrained ResNet50 is used as the encoder, separate to 4 parts
    self.resnet = models.resnet50(pretrained=pretrained)
    self.num_classes = num_classes

    # self.channels = [32, 64, 128, 256] # This is according to the paper, but does not work
    self.channels = [256, 512, 1024, 2048]

    # Pre-Encoder
    self.conv = self.resnet.conv1
    self.bn = self.resnet.bn1
    self.relu = self.resnet.relu
    self.maxpool = self.resnet.maxpool

    # Encoder
    self.encoder1 = self.resnet.layer1
    self.encoder2 = self.resnet.layer2
    self.encoder3 = self.resnet.layer3
    self.encoder4 = self.resnet.layer4

    # Decoder
    self.decoder4 = DecoderBlock(self.channels[-1])
    self.decoder3 = DecoderBlock(self.channels[-2])
    self.decoder2 = DecoderBlock(self.channels[-3])
    self.decoder1 = DecoderBlock(self.channels[-4])
    self.af3 = AFM(self.channels[-2], self.channels[-2])
    self.af2 = AFM(self.channels[-3], self.channels[-3])
    self.af1 = AFM(self.channels[-4], self.channels[-4])


    # Final Classification Head
    self.deconv_last = nn.ConvTranspose2d(self.channels[-4] // 2, 32, 3, stride=2)
    self.relu_last = nn.ReLU(inplace=True)
    self.conv_last1 = nn.Conv2d(32, 32, 3)
    self.conv_last2 = nn.Conv2d(32, self.num_classes, 2, padding=1)

  def forward(self, x):
    # Encoder
    x = self.maxpool(self.relu(self.bn(self.conv(x)))) 
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(enc1)
    enc3 = self.encoder3(enc2)
    enc4 = self.encoder4(enc3)

    # Decoder
    dec4 = self.decoder4(enc4)
    a3 = self.af3(dec4, enc3)
    dec3 = self.decoder3(a3)
    a2 = self.af2(dec3, enc2)
    dec2 = self.decoder2(a2)
    a1 = self.af1(dec2, enc1)
    dec1 = self.decoder1(a1)

    # Classifier
    logits = self.relu_last(self.deconv_last(dec1))
    logits = self.relu_last(self.conv_last1(logits))
    logits = self.conv_last2(logits)

    # Log-Softmax  
    out = F.log_softmax(logits, dim=1)
    return out