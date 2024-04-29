import torch

class Repeat(torch.nn.Module):
    r"""Repeat the entire audio _count_ times, or once if _count_ is not given.

    Args:
        count (int): The number of times to repeat the signal. (Default: ``1``)
    """
    def __init(self, count=1):
        super(Repeat, self).__init__()
        self.count = count

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): The input signal of dimension (channel, time)

        Returns:
            torch.Tensor: Output signal of dimension (channel, time)
        """
        if self.count > 0:
            return waveform.repeat(1, self.count)

        return waveform
