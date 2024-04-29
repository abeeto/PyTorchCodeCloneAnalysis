from pathlib import Path
import argparse

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from config import *
from script import train_loop, evaluate_loop
from script.utils import find_next_id, be_deterministic, defaultdict_none

be_deterministic()

parser = argparse.ArgumentParser()
# required settings
parser.add_argument("--data", type=str, help="dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--model", type=str, help="model type", choices=MODELS.keys(), required=True)
parser.add_argument("--batch-size", type=int, help="training batch size", required=True)

# hyper parameters
parser.add_argument("--lr", type=float, help="training learning rate", default=3e-4)
parser.add_argument("--alpha", type=float, help="width multiplier of MobileNet", default=1.0)
parser.add_argument("--input-resolution", type=int, help="input resolution of MobileNet", default=224)

# misc parameters
parser.add_argument("--device", type=str, help="cpu or gpu?", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--num-worker", type=int, help="sub-processes for data loading", default=0)
parser.add_argument("--num-epoch", type=int, help="training epochs", default=50)

# misc settings
parser.add_argument("--print-step", type=int, help="How often to print progress (in batch)?")
parser.add_argument("--save-step", type=int, help="How often to save network (in epoch)?")
parser.add_argument("--out-dir", type=Path, help="Where to save network (in epoch)?")
parser.add_argument("--resume", type=Path, help="path to saved network")


def train(configs):
    # hyper parameters
    batch_size = configs.batch_size
    lr = configs.lr
    alpha = configs.alpha
    input_resolution = configs.input_resolution

    # experiment settings
    data = configs.data
    model_type = configs.model
    if configs.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda is not available. Training will continue with cpu")
        device = "cpu"
    else:
        device = configs.device
    num_worker = configs.num_worker
    num_epoch = configs.num_epoch
    save_step = configs.save_step
    out_directory = configs.out_dir
    resume = configs.resume

    assert data in DATASETS.keys()
    assert model_type in MODELS.keys()

    # training configs
    c = defaultdict_none()
    c["print_step_train"] = configs.print_step
    c["device"] = device

    # select dataset
    print(f"Reading dataset {data}. This may take a while if dataset is large ...")
    Dataset = DATASETS[data]["class"]
    train_root = DATASETS[data]["train_root"]
    test_root = DATASETS[data]["test_root"]
    train_dataset = Dataset(root=train_root, is_train=True)
    test_dataset = Dataset(root=test_root, is_train=False)
    num_class = train_dataset.num_class

    # set up model, dataloader, optimizer, criterion
    Network = MODELS[model_type]
    if resume is not None:
        # ensure pretrained data don't clash with input arguments
        state = torch.load(resume)
        if state["alpha"] != alpha or state["input_resolution"] != input_resolution:
            print(f"WARNING: different model parameters. "
                  f"Model will be created with alpha={state['alpha']} and "
                  f"input_resolution={state['input_resolution']}")
        if state["num_class"] != num_class:
            raise Exception(f"Expected output features to be {num_class}, "
                            f"but pretrained model got {state['num_class']}")

        num_class, alpha, input_resolution = state["num_class"], state["alpha"], state["input_resolution"]
        network = Network(num_class, alpha=alpha, input_resolution=input_resolution).to(device)
        print(f"Loading pretrained network {network} from {resume} ...")
        network.load_state_dict(state["state_dict"])
        from_epoch = state["epoch"] + 1
    else:
        network = Network(num_class, alpha=alpha, input_resolution=input_resolution).to(device)
        print(f"Initiating network {network} ...")
        from_epoch = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(from_epoch, num_epoch):
        print(f"{'-' * 10} Epoch {epoch:2d}/{num_epoch} {'-' * 10}")
        train_loop(network, train_dataloader, optimizer, criterion, **c)

        print(f"{'-' * 5} Validation result {'-' * 5}")
        evaluate_loop(network, test_dataloader, criterion, **c)

        if save_step is not None and out_directory is not None and epoch % save_step == 0:
            out_path = Path(out_directory) / f"{find_next_id(Path(out_directory)):04d}"
            out_path.mkdir(exist_ok=True, parents=True)
            save_to = out_path / f"{model_type}-a{alpha * 100:3d}-r{input_resolution:d}-c{num_class}-e{epoch:04d}.pth"
            print(f"{'-' * 5} Saving model to {save_to} {'-' * 5}")
            state = {"epoch": epoch, "alpha": alpha, "input_resolution": input_resolution,
                     "num_class": num_class, "state_dict": network.state_dict()}
            torch.save(state, str(save_to))

        print('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
