import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.datasets import CIFAR10


from models import BaseVAE
from models.types import *


class VariationalAutoencoderExperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VariationalAutoencoderExperiment, self).__init__()
        self.model = vae_model
        self.params = params

        self.curr_device = None

        self.num_train_imgs = None
        self.num_val_imgs = None
        self.sample_dataloader = None

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def train_dataloader(self):
        transform = self.data_transform(train=True)
        if self.params['dataset'] == 'cifar10':
            dataset = CIFAR10(root=os.path.join(os.getcwd(), self.params['data_path']),
                              train=True, transform=transform, download=True)
        else:
            raise ValueError('Undefined dataset type')
        self.num_train_imgs = len(dataset)
        # TODO: add pin memory and num_workers to data loader
        return DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True,
                          drop_last=True, num_workers=self.params['num_workers'],
                          pin_memory=True)

    def val_dataloader(self):
        transform = self.data_transform(train=False)
        # TODO: add test_batch_size param to params
        if self.params['dataset'] == 'cifar10':
            data_path = os.path.join(os.getcwd(), self.params['data_path'])
            self.sample_dataloader = DataLoader(CIFAR10(data_path, train=False,
                                                        transform=transform,
                                                        download=True),
                                                batch_size=144,
                                                pin_memory=True,
                                                num_workers=self.params['num_workers'],
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        data, target = batch
        self.curr_device = data.device
        results = self.forward(data, labels=target)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        data, target = batch
        self.curr_device = data.device

        results = self.forward(data, labels=target)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size']/self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)

        vutils.save_image(test_input.cpu().data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"orig_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)
        vutils.save_image(recons.cpu().data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass
        del test_input, recons #, samples

    def configure_optimizers(self):
        optimizers = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optimizers.append(optimizer)
        return optimizers

    def data_transform(self, train=False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transform
