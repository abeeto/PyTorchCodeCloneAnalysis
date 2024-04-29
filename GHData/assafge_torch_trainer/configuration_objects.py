from dataclasses import dataclass
from typing import Dict, Any


class LossConfig:
    def __init__(self, kargs: dict, module_path: str = 'torch.nn', factor: float = 1., im_channels: list = None):
        self.module_path: str = module_path
        self.kargs: dict = kargs if kargs is not None else {}
        self.factor: float = factor
        if im_channels is not None:
            self.im_channels = slice(*im_channels)
        else:
            self.im_channels = None


@dataclass
class ModelConfig:
    path: str
    type: str
    batch_size: int
    epochs: int
    in_channels: int
    out_channels: int
    kargs: dict
    loss: Dict[str, LossConfig]
    consistency_loss: bool = False
    train_traces: Dict = None
    test_traces: Dict = None

    def __post_init__(self):
        loss_cfg = self.loss.copy()
        self.loss = {}
        for key in loss_cfg:
            self.loss[key] = LossConfig(**loss_cfg[key])


@dataclass
class DatasetConfig:
    path: str
    type: str
    kargs: dict
    loader_kargs: dict = None


@dataclass
class OptimizerConfig:
    type: str
    kargs: Dict


class TrainingConfiguration:
    def __init__(self, data, model, optimizer):
        self.data = DatasetConfig(**data)
        self.model = ModelConfig(**model)
        self.optimizer = OptimizerConfig(**optimizer)