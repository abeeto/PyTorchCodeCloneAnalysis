import numpy as np
import torch

class TensorFFT2D:
    '''
        Drop in class for usage in torchvision.transforms pipeline.
        Returns 2D FFT of passed in image.
        Assumes passed in image is of shape C*H*W.
    '''
    def __init__(self, drop_nyquist = True, amplitude_only = True, phase_only = False):
        '''
            Args:
                drop_nyquist:
                    Drops all frequency bins above Nyquist Frequency.
                    Done at last axis.
                amplitude_only:
                    Only one of this and phase_only can be True. If True,
                    returns only the amplitudes at each frequency bin.
                    If both this and phase_only are False, it returns a
                    complex Tensor containing the Fourier coefficients of
                    the image.
                phase_only:
                    Only one of this and amplitude_only can be True. If True,
                    returns only the phase shifts at each frequency bin.
                    If both this and amplitude_only are False, it returns a
                    complex Tensor containing the Fourier coefficients of
                    the image.
        '''
        assert not (amplitude_only and phase_only), "Both amplitude_only and phase_only are True, only one or less can be True at once"
        self.drop_nyquist = drop_nyquist
        self.amplitude_only = amplitude_only
        self.phase_only = phase_only

    def __call__(self, image):
        '''
            Args:
                image:
                    Tensor of shape C*H*W.
            Desc:
                Calculates and returns 2D FFT of each channel in image Tensor.
                If drop_nyquist is True, drops bins above Nyquist Frequency
                along last axis.
        '''
        image = np.fft.fft2(image)
        if self.drop_nyquist:
            image = image[:,:,:int(image.shape[-1]/2) ]
        if self.amplitude_only:
            image = np.abs(image)
        elif self.phase_only:
            image = np.angle(image)
        return image

class RandomZeroPadding:
    '''
        Drop in class for usage in torchvision.transforms pipeline.
        Returns image consisting of regular image plus random
        length zero-padding at all sides, with final shape being
        target_shape param passed in __init__.
        Can pad all channels individually as well.
    '''
    def __init__(self, target_shape, separate_channels = False):
        '''
            Args:
                target_shape:
                    Shape of output image after random padding is added.
                separate_channels:
                    If True, channels are padded separately rather than all
                    channels receiving same padding.
        '''
        self.target_shape = target_shape
        self.separate_channels = separate_channels

    def __call__(self, image):
        '''
            Args:
                image:
                    Tensor of shape C*H*W
            Desc:
                Adds zero-padding at all edges at random lengths.
                Final output shape is self.target_shape.
                If self.separate_channels, all channels padded separately.
        '''
        canvas = torch.zeros(image.shape[0], *self.target_shape)
        if self.separate_channels:
            y, x = np.random.randint(low = 0,
                                     high = [self.target_shape[0] - image.shape[1], self.target_shape[1] - image.shape[2] ],
                                     size = [2, image.shape[0] ]).T
            for channel, [y, x] in enumerate(zip(y, x) ):
                canvas[channel, y : y + image.shape[1], x : x + image.shape[2] ] = image[channel]
        else:
            y, x = np.random.randint(low = 0,
                                     high = [self.target_shape[0] - image.shape[1], self.target_shape[1] - image.shape[2] ],
                                     size = 2)
            canvas[:, y : y + image.shape[1], x : x + image.shape[2] ] = image
        return canvas
