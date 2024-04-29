import torch_resizer
import torch.nn as nn
import numpy as np


class BackProjectionLayer(nn.Module):
    def __init__(self, big_shape, scale_factor=None, small_shape=None, kernel=None, antialiasing=True):
        """
        A layer for performing back projection, given a ref small image and a current output image.
        this uses torch resizer https://github.com/assafshocher/PyTorch-Resizer
        Args:
            big_shape (numpy array): array of two numbers which are the size of the current SR output image
            scale_factor (float or numpy array, optional): the SR scale_factor which is >1 (downscaling is done by 1/scale). If not provided, calculated as ceil(out/in) (not recomended)
            small_shape (numpy array, optional): array of two numbers which are the size of the reference small image. If not provided, calculated as out/in (not recomended)
            kernel (string, optional): the method for resizing. If not provided uses the resizer default which is bicubic
            antialiasing (bool, optional): whther to peroform antialiasing on downscaling (see resizer). Defaults to True.
        """
        super(BackProjectionLayer, self).__init__()

        # downscale layer. Note that scale_factor is assumed >1, the super-resolution factor, so we use 1/scale_factor here
        self.downscale_layer = torch_resizer.Resizer(in_shape=np.concatenate([[1, 3], big_shape]),
                                                     scale_factor=1 / scale_factor,
                                                     output_shape=small_shape,
                                                     kernel=kernel,
                                                     antialiasing=antialiasing)

        # upscale is done swithcing in and out shapes and using scale_factor rather than 1/scale_factor
        self.upscale_layer = torch_resizer.Resizer(in_shape=small_shape,
                                                   scale_factor=scale_factor,
                                                   output_shape=big_shape,
                                                   kernel=kernel,
                                                   antialiasing=antialiasing)

        # define a conv layer to be used after the bp upscaling, creating a learned upsampling. this is image to image so channels in and out are 3.
        self.bp_conv = nn.Conv2d(3, 3, 7, padding=3, groups=1, bias=False)

    def forward(self, in_big_image_tensor, ref_small_image_tensor):
        # perform back-projection (downscale, subtract from small input, upscale the diff, then one conv to make the upsampling leanable)
        downscaled = self.downscale_layer(in_big_image_tensor)
        diff = ref_small_image_tensor - downscaled
        upscaled_diff = self.upscale_layer(diff)
        learned_upscaled_diff = self.bp_conv(upscaled_diff)

        # return the input image with the fix added to it
        return in_big_image_tensor + learned_upscaled_diff
