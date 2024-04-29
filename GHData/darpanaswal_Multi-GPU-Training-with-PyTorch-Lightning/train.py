import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger as wandb
import argparse
from pathlib import Path
import os

class LightningANN(pl.LightningModule):
    def __init__(self, X_size, n_hidden_nodes, y_size, args):
        super(LightningANN, self).__init__()
        self.layer1 = nn.Linear(X_size, n_hidden_nodes)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden_nodes, y_size)
        self.args = args

    def forward(self, X):
        return self.layer2(self.relu(self.layer1(X)))

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.reshape(-1, 28 * 28)

        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        return loss

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.args.batch_size, 
            shuffle=False, num_workers = 8, persistent_workers=True)

        return train_loader
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        X = X.reshape(-1, 28 * 28)

        outputs = self(X)          
        loss = F.cross_entropy(outputs, y)
        return {"test_loss": loss}

    def test_dataloader(self):
        test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.args.test_batch_size, 
            shuffle=False, num_workers = 8, persistent_workers=True)

        return test_loader
    
    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.args.lr)

wandb_logger = wandb(name='NAdam-64-1', project='LightningANN')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Lightning ANN Example')
    parser.add_argument('--batch_size', type=int, default=os.environ.get("BATCH_SIZE"), metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=os.environ.get("EPOCHS"), metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--gpu', type=int, default=os.environ.get("GPU"), metavar = 'N',
                        help='number of gpus used to train the model (default: 1')
    parser.add_argument('--lr', type=float, default=os.environ.get("LR"), metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    model = LightningANN(X_size = 784, n_hidden_nodes = 500, y_size = 10, args = args)
    if (args.gpu == 0):
        trainer = Trainer(max_epochs=args.epochs, logger= wandb_logger)
        trainer.fit(model)
    else:
        trainer = Trainer(max_epochs=args.epochs, strategy="ddp_find_unused_parameters_false",accelerator='gpu', devices = args.gpu, logger= wandb_logger)
        trainer.fit(model)

    if args.save_model:
        Path("model").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "model/LightningAnn.pt")