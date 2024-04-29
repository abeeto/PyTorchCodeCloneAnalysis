# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn
from torch.nn import functional as F
import copy
import math

__all__ = [
    "_Encoder", "_Decoder",
    "",
]


def _generate_source_mask(tensor: torch.Tensor) -> torch.Tensor:
    tensor_length = tensor.size(1)
    mask = torch.ones((tensor_length, tensor_length), device=tensor.device)

    return mask


def _generate_source_target_mask(padding_symbol: int,
                                 source_tensor: torch.Tensor,
                                 target_tensor: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    # Shape: [N, 1, source_length, 1]
    target_padding_mask = (target_tensor != padding_symbol).unsqueeze(1).unsqueeze(3)
    target_tensor_length = target_tensor.size(1)
    # Returns a lower triangular matrix below the main diagonal of a matrix
    target_sub_mask = torch.tril(torch.ones((target_tensor_length, target_tensor_length),
                                            dtype=torch.uint8, device=source_tensor.device))
    source_tensor_mask = torch.ones((target_tensor_length, source_tensor.size(1)),
                                    dtype=torch.uint8, device=source_tensor.device)
    target_tensor_mask = target_padding_mask & target_sub_mask.bool()

    return source_tensor_mask, target_tensor_mask


class _MultiInputSequential(nn.Sequential):
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        for index, module in enumerate(self):
            if index == 0:
                x = module(*x)
            else:
                x = module(x)
        return x


class _MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, multi_attention_heads: int, dimensions: int) -> None:
        """Initialize the multi-head attention mechanism function

        Args:
            multi_attention_heads (int): Number of multi-head attention units
            dimensions (int): Model dimension

        """
        super(_MultiHeadAttention, self).__init__()

        # How many attention heads in each dimension
        self.d_k = int(dimensions / multi_attention_heads)
        self.heads = multi_attention_heads
        # (q, k, v, last output layer)
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(dimensions, dimensions) for _ in range(4))])

    @torch.jit.script_method
    def dot_product_attention(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """Compute `Scaled Dot Product Attention`
        
        Args:
            query (torch.Tensor): Tensor vector. Shape: [N, heads, sequence_length, d_q]
            key (torch.Tensor): Tensor vector. [N, heads, sequence_length, d_k]
            value (torch.Tensor): Tensor vector. [N, heads, sequence_length, d_v]
            mask (torch.Tensor): Mask tensor vector. [N, 1, sequence_length, sequence_length]

        Returns:
            attention_feature (torch.Tensor) Attention mask feature. Shape: Shape: [N, head, sequence_length, d_v]

        """
        d_k = value.size(-1)
        # mask_score [N, heads, sequence_length, sequence_length]
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # If the mask matrix is not set, a smaller 2D matrix is defined
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # A mask matrix with all values 0-1
        attention_score = F.softmax(score, dim=-1)

        # Attention features. Shape: [N, head, sequence_length, d_v]
        attention_feature = torch.matmul(attention_score, value)

        return attention_feature

    @torch.jit.script_method
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # how many tensor vectors
        batch_size = query.size(0)

        # Calculate the projection value in the attention mechanism
        query, key, value = [linear(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) for linear, x in
                             zip(self.linears, (query, key, value))]

        # Focus attention on all projected vectors
        dot_product_attention = self.dot_product_attention(query, key, value, mask=mask)

        # Linear fusion of projected attention features. Shape: [N, sequence_length, d_k / head]
        dot_product_attention = dot_product_attention.transpose(1, 2).contiguous().view(batch_size,
                                                                                        -1,
                                                                                        self.heads * self.d_k)
        # Shape: [N, sequence_length, d_k / head]
        out = self.linears[-1](dot_product_attention)

        return out


class _PositionWiseFeedForward(nn.Module):
    def __init__(self, dimensions: int, feed_forward_dimensions: int, dropout_ratio: float):
        """

        Args:
            dimensions (int): Tensor vector
            feed_forward_dimensions(int): Feed forward tensor vector
            dropout_ratio (float): probability of an element to be zeroed

        """
        super(_PositionWiseFeedForward, self).__init__()
        self.position_wise_feed_forward_block = nn.Sequential(
            nn.Linear(dimensions, feed_forward_dimensions),
            nn.ReLU(True),
            nn.Dropout(dropout_ratio, True),
            nn.Linear(feed_forward_dimensions, dimensions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.position_wise_feed_forward_block(x)

        return out


class _PositionEncoding(torch.jit.ScriptModule):
    def __init__(self, dimensions: int, dropout_ratio: float, max_length: int) -> torch.Tensor:
        """

        Args:
            dimensions (int): Tensor vector
            dropout_ratio (float): probability of an element to be zeroed
            max_length (int): Tensor vector maximum length

        """
        super(_PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio, True)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_length, dimensions)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dimensions, 2).float() * -(math.log(10000.0) / dimensions))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.pe[:, :x.size(1)]
        out = self.dropout(out)

        return out


class _MultiAspectGCAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels_growth_ratio: float,
                 multi_attention_heads: int) -> None:
        """Implement Multi-Aspect Global Context Attention

        Args:
            in_channels (int): Number of input image channels
            channels_growth_ratio (int): Image channel number growth rate
            multi_attention_heads (int): Number of multi-head attention units

        """
        super(_MultiAspectGCAttention, self).__init__()
        self.multi_attention_heads = multi_attention_heads
        self.in_channels = in_channels

        self.single_attention_header_channels = int(in_channels / multi_attention_heads)

        self.conv_mask = nn.Conv2d(self.single_attention_header_channels, 1, (1, 1))
        self.softmax = nn.Softmax(dim=2)
        self.channel_concat_conv = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels * channels_growth_ratio), (1, 1)),
            nn.LayerNorm([int(in_channels * channels_growth_ratio), 1, 1]),
            nn.ReLU(True),
            nn.Conv2d(int(in_channels * channels_growth_ratio), in_channels, (1, 1)),
        )
        self.concat_conv = nn.Conv2d(int(2 * in_channels), in_channels, (1, 1))

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = x.size()

        # Shape: [N * multi_attention_heads, C, H , W]
        x = x.view(batch_size * self.multi_attention_heads,
                   self.single_attention_header_channels,
                   height,
                   width)
        input_x = x

        # Shape: [N * multi_attention_heads, C, H * W]
        input_x = input_x.view(batch_size * self.multi_attention_heads,
                               self.single_attention_header_channels,
                               height * width)

        # Shape: [N * multi_attention_heads, 1, C, H * W]
        input_x = input_x.unsqueeze(1)

        # Shape: [N * multi_attention_heads, 1, H, W]
        context_mask = self.conv_mask(x)

        # Shape: [N * multi_attention_heads, 1, H * W]
        context_mask = context_mask.view(batch_size * self.multi_attention_heads, 1, height * width)

        # Shape: [N * multi_attention_heads, 1, H * W]
        context_mask = self.softmax(context_mask)

        # Shape: [N * multi_attention_heads, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N * multi_attention_heads, 1, C, H * W] * [N * multi_attention_heads, 1, H * W, 1]
        # Shape: [N * multi_attention_heads, 1, C, 1]
        context = torch.matmul(input_x, context_mask)

        # [N, multi_attention_heads * C, 1, 1]
        context = context.view(batch_size, self.multi_attention_heads * self.single_attention_header_channels, 1, 1)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x

        # [N, C, 1, 1]
        channel_concat_term = self.channel_concat_conv(context)

        # use concat
        _, _, height, width = out.shape

        out = torch.cat([out, channel_concat_term.expand(-1, -1, height, width)], dim=1)
        out = self.concat_conv(out)
        out = F.layer_norm(out, [self.in_channels, height, width])
        out = F.relu(out, True)

        return out


class _ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 channels_growth_ratio: float,
                 multi_attention_heads: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 use_magca: bool = False) -> None:
        """Residual network basic block

        Args:
            in_channels (int): The number of input channels
            mid_channels (int): The number of middle channels
            channels_growth_ratio (int): Image channel number growth rate
            multi_attention_heads (int): Number of multi-head attention units
            stride (int): Convolution operation stride. Default: 1
            use_magca (bool): Whether to use MultiAspectGCAttention module. Default: ``False``

        """
        super(_ResNetBlock, self).__init__()
        self.resnet_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (3, 3), (stride, stride), (1, 1), bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, (3, 3), (stride, stride), (1, 1), bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.9),
        )
        self.downsample = downsample
        self.stride = stride
        self.use_magca = use_magca

        self.magca = _MultiAspectGCAttention(mid_channels, channels_growth_ratio, multi_attention_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.resnet_block(x)

        if self.use_magca:
            out = self.magca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = F.relu(out, True)

        return out


class _ResNet50(nn.Module):

    def __init__(self, block: nn.Module = _ResNetBlock, layers: list = None) -> None:
        super(_ResNet50, self).__init__()

        self.channels = 64
        if layers is None:
            layers = [1, 2, 5, 3]

        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)

        self.maxpool1 = nn.MaxPool2d((2, 2), (2, 2))

        self.layer1 = self._make_layer(block, 256, layers[0], 1, False)

        self.conv3 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)

        self.maxpool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.layer2 = self._make_layer(block, 256, layers[1], 1, True)

        self.conv4 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(True)

        self.maxpool3 = nn.MaxPool2d((2, 1), (2, 1))

        self.layer3 = self._make_layer(block, 512, layers[2], 1, True)

        self.conv5 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        self.layer4 = self._make_layer(block, 512, layers[3], 1, True)

        self.conv6 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(True)

        # Initialize neural network weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int, use_magca: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, (1, 1), (stride, stride), bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, use_magca)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        layers = nn.Sequential(*layers)

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        out = self.layer1(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool2(out)
        out = self.layer2(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool3(out)
        out = self.layer3(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.layer4(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)

        return out


class _ConvEmbeddingGC(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ResNet50()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(x)

        # Shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = feature.shape
        feature = feature.view(batch_size, channels, height * width)
        feature = feature.permute((0, 2, 1))

        return feature


class _Encoder(nn.Module):
    def __init__(self, dimensions: int, dropout_ratio: float, max_length: int) -> torch.Tensor:
        """Implement the encoder in the Transform structure

        Args:
            dimensions (int): Tensor vector
            dropout_ratio (float): probability of an element to be zeroed
            max_length (int): Tensor vector maximum length

        """
        super(_Encoder, self).__init__()
        self.position = _PositionEncoding(dimensions, dropout_ratio, max_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.position(x)

        return output


class _Decoder(nn.Module):
    def __init__(self, multi_attention_heads: int,
                 dimensions: int,
                 feed_forward_dimensions: int,
                 dropout_ratio: float,
                 num_classes: int,
                 padding_symbol: int,
                 layer_number: int):
        """Implement the decoder in the Transform structure

        Args:
            multi_attention_heads (int): Number of multi-head attention units
            dimensions (int): Model dimension
            feed_forward_dimensions(int): Feed forward tensor vector
            dropout_ratio (float): probability of an element to be zeroed
            num_classes (int): The number of output categories of the feature layer
            padding_symbol (int): Filling all around
            layer_number (int): How many layers of cyclic structures are included in the decoder
        """
        super(_Decoder, self).__init__()
        self.attention = nn.Sequential(
            _MultiHeadAttention(multi_attention_heads, dimensions),
            _MultiHeadAttention(multi_attention_heads, dimensions),
            _MultiHeadAttention(multi_attention_heads, dimensions),
        )
        self.attention = nn.ModuleList(
            [_MultiHeadAttention(multi_attention_heads, dimensions, dropout_ratio) for _ in range(layer_number)]
        )
        self.source_attention = nn.ModuleList(
            [_MultiHeadAttention(multi_attention_heads, dimensions, dropout_ratio) for _ in range(layer_number)]
        )
        self.position_feed_forward = nn.ModuleList([
            _PositionWiseFeedForward(dimensions, feed_forward_dimensions, dropout_ratio) for _ in range(layer_number)]
        )
        self.position = _PositionEncoding(dimensions, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio, True)
        self.layer_norm = nn.LayerNorm(dimensions, 1e-6)
        self.embedding = nn.Embedding(num_classes, dimensions)
        self.sqrt_dimensions = math.sqrt(dimensions)
        self.padding_symbol = padding_symbol
        self.layer_number = layer_number

    def forward(self, target, encode_stage_result):
        target_embedding = self.embedding(target) * self.sqrt_dimensions
        target_position = self.position(target_embedding)
        source_mask, target_mask = _generate_source_target_mask(self.padding_symbol, encode_stage_result, target)

        output = target_position
        for i in range(self.layer_number):
            layer_norm_output = self.layer_norm(output)
            layer_norm_output_dropout = self.dropout(self.attention[i](layer_norm_output,
                                                                       layer_norm_output,
                                                                       layer_norm_output,
                                                                       target_mask))
            output = torch.add(output, layer_norm_output_dropout)

            layer_norm_output = self.layer_norm(output)
            layer_norm_output_dropout = self.dropout(self.source_attention[i](layer_norm_output,
                                                                              encode_stage_result,
                                                                              encode_stage_result,
                                                                              source_mask))

            output = torch.add(output, layer_norm_output_dropout)

            layer_norm_output = self.layer_norm(output)
            layer_norm_output_dropout = self.dropout(self.position_feed_forward[i](layer_norm_output))

            output = torch.add(output, layer_norm_output_dropout)

        output = self.layer_norm(output)

        return output


class MASTER(nn.Module):

    def __init__(self, num_classes: int):
        super(MASTER, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 16 * 64

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 8 * 32

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 4 x 16

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 2 x 16

            nn.Conv2d(512, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # image size: 1 x 16
        )

        self.recurrent_layers = nn.Sequential(
            _BidirectionalLSTM(512, 256, 256),
            _BidirectionalLSTM(256, 256, num_classes),
        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Feature sequence
        features = self.convolutional_layers(x)  # [b, c, h, w]
        features = features.squeeze(2)  # [b, c, w]
        features = features.permute(2, 0, 1)  # [w, b, c]

        # Deep bidirectional LSTM
        out = self.recurrent_layers(features)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
