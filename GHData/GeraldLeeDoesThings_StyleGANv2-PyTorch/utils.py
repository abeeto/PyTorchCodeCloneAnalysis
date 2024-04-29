import torch


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation"""
    # TODO implement as named function
    return a + (b - a) * t
