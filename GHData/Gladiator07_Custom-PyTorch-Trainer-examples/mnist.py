import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags


# custom modules
from src.trainer import Trainer, TrainerArguments
from src.utils import set_seed

config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_bool("wandb_enabled", default=True, help="enable Weights & Biases logging")

FLAGS = flags.FLAGS
FLAGS(sys.argv)   # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx
cfg = FLAGS.config
wandb_enabled = FLAGS.wandb_enabled

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, image, label=None):
        x = image
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)

        if label is not None:
            loss = nn.CrossEntropyLoss()(out, label)

            return out, loss
        return out

class MNISTDataset:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        return {
            "image": img,
            "label": label
        }

def main():
    set_seed(42)

    # init 🤗 accelerator
    accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=cfg.trainer_args['mixed_precision'],
            gradient_accumulation_steps=cfg.trainer_args['gradient_accumulation_steps'],
            log_with="wandb" if wandb_enabled else None,
        )

    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="Custom_Accelerate_Trainer_Tests", config=cfg.to_dict())

    # Load Data
    train_dataset = MNISTDataset(datasets.MNIST(root = "dataset/MNIST", train=True, transform=transforms.ToTensor(), download=True))
    train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.batch_size, shuffle=True)
    accelerator.print(f"Train Loader = {train_loader}")

    val_dataset = MNISTDataset(datasets.MNIST(root = "dataset/MNIST", train=False, transform=transforms.ToTensor(), download=True))
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.batch_size, shuffle=False)


    # Initialize the network
    model = CNN(in_channels=cfg.in_channels, num_classes=cfg.num_classes)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # implement custom metrics function
    # these metrics will be calculated and logged after every evaluation phase
    # do all the post-processing on logits in this function
    def compute_metrics(logits, labels):
        preds = np.argmax(logits, axis=1)
        acc_score = accuracy_score(labels, preds)
        # implement any other metrics you want and return them in the dict format
        return {"accuracy": acc_score}
    # trainer args
    args = TrainerArguments(**cfg.trainer_args)

    # initialize my custom trainer
    trainer = Trainer(
        model=model,
        args=args,
        optimizer=optimizer,
        accelerator=accelerator,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        compute_metrics=compute_metrics,
    )

    # call .fit to perform training + validation
    trainer.fit()

    # predict on test set if you have one
    # preds = trainer.predict("./outputs/best_model.bin", test_dataloader)
    
    # here predicting on val dataset to demonstrate the `.predict()` method
    val_preds = trainer.predict("./outputs/best_model.bin", val_loader)
    if wandb_enabled:
        # end trackers
        accelerator.end_training()


if __name__ == "__main__":
    main()
