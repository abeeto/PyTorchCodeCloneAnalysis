"""
Configuration file for Torch Hub
"""
dependencies = [
    "torch",
    "requests",
    "tqdm",
]

import typing as _t
from raft_lib.wrapper import RAFTWrapper as _RAFTWrapper


def raft(
    pretrained: bool = False,
    dataset: _t.Literal["kitti", "sintel", "chairs", "small", "things"] = "sintel",
    mixed_precision: bool = False,
):
    """
    RAFT model as in https://github.com/princeton-vl/RAFT, with the
    same interface.

    Parameters
    ----------
    pretrained: bool
        if true loads the official trained models from https://github.com/princeton-vl/RAFT
    dataset: kitti | sintel | chairs | small | things
        the dataset on which the pretrained model has been trained, default sintel

    Returns
    -------
    out: nn.Model
        the RAFT model

    Examples
    --------
    To retrieve the optical flow

    >>> model = raft(pretrained=True, dataset="kitti")
    >>> flow, flow_up = model(img_1, img_2, iters=12, flow_init=None, test_mode=True)
    """

    model = _RAFTWrapper(
        dataset if pretrained else None, mixed_precision=mixed_precision
    )
    return model
