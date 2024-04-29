
import torch
import torch.nn as nn

"""
initial
"""


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super(InitialBlock, self).__init__()
        if (relu):
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        # maini branch
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=bias)
        # another branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_relu = activation()

    def forward(self, x):
        x1 = self.main_branch(x)
        x2 = self.ext_branch(x)
        out = torch.cat((x1, x2), 1)
        out = self.bn(out)

        return self.out_relu(out)


"""
Bottleneck with downsample
"""
class Bottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()
        """
        internal_ratio check
        """
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))
        internal_channels = channels // internal_ratio

        if (relu):
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        """
        Main branch first 1x1
        """
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation())
        """
        using symmetric
        """
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation())
        """
        1x1
        """
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation())
        """
        regu
        """
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        """
        activation
        """
        self.out_activation = activation()

    def forward(self, x):
        main = x
        # print(type(x))
        # print("==========")
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        out = main + ext
        return self.out_activation(out)


"""
Bottleneck with downsample
"""


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, return_indices=False, dropout_prob=0, bias=False,
                 relu=True):
        super(DownsamplingBottleneck, self).__init__()

        self.return_indices = return_indices
        """
        internal_ratio check
        """
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio

        if (relu):
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        """
        MaxPool2d
        """
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        """
        2x2 2 downsample
        """
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation())

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation())

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        if (self.return_indices):
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)
        # Add main and extension branches
        out = main + ext
        return self.out_activation(out), max_indices


"""
Bottleneck with upsampling
"""


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, bias=False, relu=True):
        super(UpsamplingBottleneck, self).__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        """
        Transposed convolution
        """
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)

        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):
    def __init__(self):
        super(ENet, self).__init__()
        binary_seg=2
        embedding_dim=5
        num_classes=8
        encoder_relu = False
        decoder_relu = True

        ## init
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder -share
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)

        self.regular1_1 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = Bottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                        relu=encoder_relu)

        self.dilated2_4 = Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)

        self.asymmetric2_7 = Bottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                        relu=encoder_relu)
        self.dilated2_8 = Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder -for binary
        self.b_regular3_0 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.b_dilated3_1 = Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.b_asymmetric3_2 = Bottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                        relu=encoder_relu)

        self.b_dilated3_3 = Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.b_regular3_4 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.b_dilated3_5 = Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.b_asymmetric3_6 = Bottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                        relu=encoder_relu)
        self.b_dilated3_7 = Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder -for embedded
        self.e_regular3_0 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.e_dilated3_1 = Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.e_asymmetric3_2 = Bottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                        relu=encoder_relu)

        self.e_dilated3_3 = Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.e_regular3_4 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.e_dilated3_5 = Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.e_asymmetric3_6 = Bottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                        relu=encoder_relu)
        self.e_dilated3_7 = Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # binary branch
        self.upsample_binary_4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_4_1 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_4_2 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample_binary_5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_5_1 = Bottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.binary_transposed_conv = nn.ConvTranspose2d(16, binary_seg, kernel_size=3, stride=2, padding=1, bias=False)

        # embedding branch
        self.upsample_embedding_4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_4_1 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_4_2 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample_embedding_5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_5_1 = Bottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.embedding_transposed_conv = nn.ConvTranspose2d(16, embedding_dim, kernel_size=3, stride=2, padding=1,
                                                            bias=False)

    def forward(self, x):
        # TODO
        # Initial block
        ##256x512
        input_size = x.size()

        ##batch_size, 16, 128x256
        x = self.initial_block(x)

        # Stage 1 - Encoder-share
        ##64x128
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        #->2,64,64,128
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder -share
        ##2,128,32,64
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        ##2,128, 32x64
        b_x = self.b_regular3_0(x)
        b_x = self.b_dilated3_1(b_x)
        b_x = self.b_asymmetric3_2(b_x)
        b_x = self.b_dilated3_3(b_x)
        b_x = self.b_regular3_4(b_x)
        b_x = self.b_dilated3_5(b_x)
        b_x = self.b_asymmetric3_6(b_x)
        b_x = self.b_dilated3_7(b_x)

        e_x = self.e_regular3_0(x)
        e_x = self.e_dilated3_1(e_x)
        e_x = self.e_asymmetric3_2(e_x)
        e_x = self.e_dilated3_3(e_x)
        e_x = self.e_regular3_4(e_x)
        e_x = self.e_dilated3_5(e_x)
        e_x = self.e_asymmetric3_6(e_x)
        e_x = self.e_dilated3_7(e_x)

        # binary branch 2,64,64,128
        x_binary = self.upsample_binary_4_0(b_x, max_indices2_0, output_size=stage2_input_size)
        x_binary = self.regular_binary_4_1(x_binary)
        x_binary = self.regular_binary_4_2(x_binary)
        x_binary = self.upsample_binary_5_0(x_binary, max_indices1_0, output_size=stage1_input_size)# 2,16,128,256
        x_binary = self.regular_binary_5_1(x_binary)
        binary_final_logits = self.binary_transposed_conv(x_binary, output_size=input_size)#2,1,256,512

        # embedding branch
        x_embedding = self.upsample_embedding_4_0(e_x, max_indices2_0, output_size=stage2_input_size)
        x_embedding = self.regular_embedding_4_1(x_embedding)
        x_embedding = self.regular_embedding_4_2(x_embedding)
        x_embedding = self.upsample_embedding_5_0(x_embedding, max_indices1_0, output_size=stage1_input_size)
        x_embedding = self.regular_embedding_5_1(x_embedding)
        instance_notfinal_logits = self.embedding_transposed_conv(x_embedding, output_size=input_size)

        return binary_final_logits, instance_notfinal_logits
