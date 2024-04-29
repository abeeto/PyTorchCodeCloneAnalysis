#  Image Segmentation

# Explore the U-Net Architecture
# Perform upscaling

from torch import ones
import torch.nn as nn
from torch.nn import ConvTranspose2d

m_type_ConvTranspose2d = ConvTranspose2d(
    1,
    1,

    kernel_size = (
        2,
        2
    ),

    stride = 2,
    padding = 0
)

input_type_Tensor = ones(
    1,
    1,
    3,
    3
)

output_type_Tensor = m_type_ConvTranspose2d( input_type_Tensor )

print( output_type_Tensor.shape )