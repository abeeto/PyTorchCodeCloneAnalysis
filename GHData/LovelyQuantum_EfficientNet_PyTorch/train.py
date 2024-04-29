from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import copy
from efficientnet_pytorch import EfficientNet
import argparse
import re
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_args():
    parser = argparse.ArgumentParser("BEiT pre-training script", add_help=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--model_name", default="efficientnet-b7", type=str)
    parser.add_argument("--load_weight", type=str)
    parser.add_argument("--save_all_models", action="store_true")
    parser.add_argument("--quiet_mode", action="store_true")
    return parser.parse_args()


def init_dataset(dataset_path, batch_size, input_size=224):
    logging.info("Initializing Datasets and Dataloaders...")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(35),
                transforms.RandomGrayscale(),
                # transforms.RandomAffine(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x])
        for x in ["train", "val"]
    }
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    num_classes = len(image_datasets["train"].classes)
    logging.info("Done.")
    return dataloaders_dict, num_classes


def init_model():
    logging.info("Loading model...")
    with HiddenPrints():
        model = EfficientNet.from_pretrained(opts.model_name)
    model._fc = nn.Linear(2560, num_classes, bias=True)
    model._fc.in_features
    logging.info("Done.")
    return model


def freeze_parameters(model, layer=0):
    logging.info("Freezing parameters...")
    params_to_update = [
        "_conv_head.weight",
        "_bn1.weight",
        "_bn1.bias",
        "_fc.weight",
        "_fc.bias",
    ]
    changed_params = []

    # Patten used to match layer numbers
    patten = re.compile(r"blocks\.[0-9]+")
    layer_num = patten.search(r"_blocks.49._expand_conv").group()
    layer_num = int(layer_num.split(".")[1])

    for name, param in model.named_parameters():
        layer_num = 0
        if patten.search(name):
            layer_num = patten.search(name).group()
            layer_num = int(layer_num.split(".")[1])
        if name in params_to_update or layer_num >= layer:
            changed_params.append(param)
        else:
            param.requires_grad = False
    logging.info("Done.")
    return changed_params


def train_model(
    model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == "val" and epoch_acc > 0.93:
                torch.save(model.state_dict(), f"num_epochs{epoch}_acc{epoch_acc}.pth")

            logging.info(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

    logging.info(f'{"#" * 10}Result{"#" * 10}')
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":
    opts = get_args()
    dataloaders_dict, num_classes = init_dataset(opts.dataset, opts.batch_size)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = init_model().to(device)
    changed_params = freeze_parameters(model)
    optimizer = optim.SGD(changed_params, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model, hist = train_model(
        model, dataloaders_dict, criterion, optimizer, num_epochs=opts.num_epochs
    )
    torch.save(model.state_dict(), opts.model_save_path)
