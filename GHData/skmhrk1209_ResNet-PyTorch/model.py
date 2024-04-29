import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, projection, num_groups):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels
        )
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)

        self.act1 = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True
        )
        nn.init.kaiming_normal_(
            tensor=self.conv1.weight,
            a=0.0,
            nonlinearity="relu"
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        self.act2 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )
        nn.init.kaiming_normal_(
            tensor=self.conv2.weight,
            a=0.0,
            nonlinearity="relu"
        )

        if projection:
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )
            nn.init.kaiming_normal_(
                tensor=self.projection.weight,
                a=0.0,
                nonlinearity="relu"
            )
        else:
            self.projection = None

    def forward(self, inputs):

        shortcut = inputs
        inputs = self.norm1(inputs)
        inputs = self.act1(inputs)
        if self.projection:
            shortcut = self.projection(inputs)
        inputs = self.conv1(inputs)

        inputs = self.norm2(inputs)
        inputs = self.act2(inputs)
        inputs = self.conv2(inputs)

        inputs += shortcut

        return inputs


class ResNet(nn.Module):

    def __init__(self, conv_param, pool_param, residual_params, num_classes, num_groups):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=conv_param.in_channels,
            out_channels=conv_param.out_channels,
            kernel_size=conv_param.kernel_size,
            stride=conv_param.stride,
            padding=conv_param.kernel_size // 2,
            bias=True
        )
        nn.init.kaiming_normal_(
            tensor=self.conv.weight,
            a=0.0,
            nonlinearity="relu"
        )

        self.pool = nn.MaxPool2d(
            kernel_size=pool_param.kernel_size,
            stride=pool_param.stride,
            padding=pool_param.kernel_size // 2
        )

        residual_blocks = []
        for residual_param in residual_params:
            residual_blocks.append(ResidualBlock(
                in_channels=residual_param.in_channels,
                out_channels=residual_param.out_channels,
                kernel_size=residual_param.kernel_size,
                stride=residual_param.stride,
                projection=True,
                num_groups=num_groups
            ))
            for _ in range(1, residual_param.blocks):
                residual_blocks.append(ResidualBlock(
                    in_channels=residual_param.out_channels,
                    out_channels=residual_param.out_channels,
                    kernel_size=residual_param.kernel_size,
                    stride=1,
                    projection=False,
                    num_groups=num_groups
                ))
        self.residual_blocks = nn.ModuleList(residual_blocks)

        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=residual_params[-1].out_channels
        )
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

        self.act = nn.ReLU()

        self.linear = nn.Linear(
            in_features=residual_params[-1].out_channels,
            out_features=num_classes,
            bias=True
        )
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):

        inputs = self.conv(inputs)
        inputs = self.pool(inputs)

        for residual_block in self.residual_blocks:
            inputs = residual_block(inputs)
        inputs = self.norm(inputs)
        inputs = self.act(inputs)

        inputs = inputs.mean((2, 3))
        inputs = self.linear(inputs)

        return inputs
