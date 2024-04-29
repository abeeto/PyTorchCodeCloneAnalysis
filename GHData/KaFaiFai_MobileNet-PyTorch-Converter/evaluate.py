"""
original preprocessing method is same as Inception, which takes 4×3×6×2 = 144 crops per image
we only take 1 center crop after resizing
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models

import argparse
from pathlib import Path

from config import *
from script import evaluate_loop
from script.utils import be_deterministic, defaultdict_none

be_deterministic()

parser = argparse.ArgumentParser()
# required settings
parser.add_argument("--data", type=str, help="dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--model", type=str, help="model type", choices=MODELS.keys(), required=True)
parser.add_argument("--pretrained-model", type=Path, help="path to pretrained model", required=True)
parser.add_argument("--batch-size", type=int, help="training batch size", required=True)

# misc parameters
parser.add_argument("--device", type=str, help="cpu or gpu?", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--num-worker", type=int, help="sub-processes for data loading", default=0)

# misc settings
parser.add_argument("--print-step", type=int, help="How often to print progress (in batch)?")


def evaluate(configs):
    # hyper parameters
    batch_size = configs.batch_size

    # experiment settings
    data = configs.data
    model_type = configs.model
    if configs.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda is not available. Testing will continue with cpu")
        device = "cpu"
    else:
        device = configs.device
    num_worker = configs.num_worker
    pretrained_model = configs.pretrained_model

    assert data in DATASETS.keys()
    assert model_type in MODELS.keys()

    # training configs
    c = defaultdict_none()
    c["print_step_test"] = configs.print_step
    c["device"] = device

    # select dataset
    print(f"Reading dataset {data}. This may take a while if dataset is large ...")
    Dataset = DATASETS[data]["class"]
    test_root = DATASETS[data]["test_root"]
    test_dataset = Dataset(root=test_root, is_train=False)
    num_class = test_dataset.num_class

    # set up model, dataloader, criterion
    Network = MODELS[model_type]
    state = torch.load(pretrained_model)
    if num_class != state["num_class"]:
        raise Exception(f"Expected output features to be {num_class}, "
                        f"but pretrained model got {state['num_class']}")

    num_class, alpha, input_resolution = state["num_class"], state["alpha"], state["input_resolution"]
    network = Network(num_class, alpha=alpha, input_resolution=input_resolution).to(device)
    print(f"Loading pretrained network {network} from {pretrained_model} ...")
    network.load_state_dict(state["state_dict"])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker)
    criterion = nn.CrossEntropyLoss()

    # testing loop
    print(f"{'-' * 5} Test result {'-' * 5}")
    evaluate_loop(network, test_dataloader, criterion, **c)


if __name__ == '__main__':
    args = parser.parse_args()
    evaluate(args)
