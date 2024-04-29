"""
all converters:
- TensorFlow: official tensorflow pretrained model -> MobileNet
- wjc852456: wjc852456's repo https://github.com/wjc852456/pytorch-mobilenet-v1 -> MobileNet
- PyTorch: official pytorch pretrained model -> MobileNetV2
- TensorFlowV2: official tensorflow pretrained model -> MobileNetV2
"""

from torchinfo import summary
from model import *
from script.utils import *
from converter import *


def main():
    state = torch.load(r"pretrained/pt-mobilenetv2-a100-r224-c1000-e0000.pth")
    num_class, alpha, input_resolution = state["num_class"], 1.00, 224
    network = MobileNetV2(num_class)
    # peek_pytorch_network(network)
    summary(network, (1, 3, input_resolution, input_resolution))
    # peek_tensorflow_network()


if __name__ == '__main__':
    main()
