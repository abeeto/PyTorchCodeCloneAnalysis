import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform


print('Into Model')

PRINTLOG = False
POOLING = False


class Logs:
    def __init__(self, printlogs=False, pooling=False):
        global PRINTLOG, POOLING
        PRINTLOG = printlogs
        POOLING = pooling

    def __str__(self):
        return f"Printing Generator logs : {PRINTLOG}, Pooling logs : {POOLING}"


class Convblock(nn.Module):

    def __init__(self, input_channel, output_channel, kernal=3, stride=1, padding=1, upconve=True):

        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernal, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),


        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class UNet(nn.Module):

    def __init__(self, input_channel, retain=True):

        super().__init__()

        self.conv1 = Convblock(input_channel, 32)
        self.conv1_1 = Convblock(32, 32)
        self.conv2 = Convblock(32, 64)
        self.conv3 = Convblock(64, 128)
        self.conv3_3 = Convblock(128, 128)
        self.conv4 = Convblock(128, 256)
        self.conv4_4 = Convblock(256, 256)
        self.neck = nn.Conv2d(256, 512, 3, 1)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 0, 1)
        self.dconv4 = Convblock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 0, 1)
        self.dconv3 = Convblock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 0, 1)
        self.dconv2 = Convblock(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 0, 1)
        self.dconv1 = Convblock(64, 32)
        self.out = nn.ConvTranspose2d(32, 1, 4, 2, 0, 1)
        self.retain = retain

    def forward(self, x):

        if PRINTLOG:
            print('Lay 0 : ', x.shape)
        conv1 = self.conv1(x)

        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)

        if PRINTLOG:
            print('Lay 1 : ', pool1.shape)
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)

        if PRINTLOG:
            print('Lay 2 : ', pool2.shape)
        conv3 = self.conv3(pool2)

        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)

        if PRINTLOG:
            print('Lay 3 : ', pool3.shape)
        conv4 = self.conv4(pool3)

        pool4 = F.max_pool2d(conv4, kernel_size=2, stride=2)
        if PRINTLOG:
            print('Lay 4 : ', pool4.shape)

        neck = self.neck(pool4)
        if PRINTLOG:
            print('Neck  : ', neck.shape)

        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4, upconv4)

        dconv4 = self.dconv4(torch.cat([upconv4, croped], 1))
        if PRINTLOG:
            print('Conv 4 : ', dconv4.shape)

        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3, upconv3)

        dconv3 = self.dconv3(torch.cat([upconv3, croped], 1))
        if PRINTLOG:
            print('Conv 3 : ', dconv3.shape)

        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2, upconv2)

        dconv2 = self.dconv2(torch.cat([upconv2, croped], 1))
        if PRINTLOG:
            print('Conv 2 : ', dconv2.shape)

        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1, upconv1)

        dconv1 = self.dconv1(torch.cat([upconv1, croped], 1))
        if PRINTLOG:
            print('Conv 1 : ', dconv1.shape)

        out = self.out(dconv1)
        if PRINTLOG:
            print('Conv 0 : ', out.shape)
        if self.retain == True:
            out = F.interpolate(out, list(x.shape)[2:])
        if PRINTLOG:
            print('Out : ', out.shape)

        return F.sigmoid(out)

    def crop(self, input_tensor, target_tensor):

        _, _, H, W = target_tensor.shape
        return transform.CenterCrop([H, W])(input_tensor)
