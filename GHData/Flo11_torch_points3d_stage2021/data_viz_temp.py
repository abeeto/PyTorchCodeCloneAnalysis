from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.datasets.base_dataset import BaseDataset
from functools import partial
import torch
import numpy as np


import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    _dataset: BaseDataset = instantiate_dataset(cfg.data)

    data_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "total": 0}
    percent_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}

    for elem in _dataset.train_dataset._datas:

        uni, coun = np.unique(elem.y, return_counts=True)
        data_dict["total"] += elem.y.shape[0]
        for val in uni:
            indice = np.where(uni == val)[0][0]
            data_dict[str(val)] += coun[indice]
    print(data_dict)

    for key, val in data_dict.items():
        percent_dict[key] = val / data_dict['total'] * 100
    print(percent_dict)

    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()