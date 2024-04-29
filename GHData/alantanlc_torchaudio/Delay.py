import torch

class Delay(torch.nn.Module):
    r"""Delay one or more audio channels such that they start at the given _position_. For example, [1.5, '+1', '3000s'] delays the first channel by 1.5 seconds, the second channel by 2.5 seconds (one second more than the previous channel), the third channel by 3000 samples, and leave any other channels that may be present un-delayed.

    Args:
        count (int): The original frequency of the signal. (Default: ``1``)
    """
    def __init(self, delay):
        super(Repeat, self).__init__()
        self.delay = delay

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): The input signal of dimension (channel, time)

        Returns:
            torch.Tensor: Output signal of dimension (channel, time)
        """
        for i in self.delay:
            

        return waveform
