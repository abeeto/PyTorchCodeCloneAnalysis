import math
import numpy as np
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10, CelebA, MNIST
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        if len(labels.shape) == 1:
            if self.params['dataset'] == 'wikiart':
                num_classes = 27
            else:
                num_classes = -1
            labels = torch.nn.functional.one_hot(labels, num_classes)
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        if len(labels.shape) == 1:
            if self.params['dataset'] == 'wikiart':
                num_classes = 27
            else:
                num_classes = -1
            labels = torch.nn.functional.one_hot(labels, num_classes)
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        # self.logger.experiment.log({'val_loss': val_loss})
        # self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        self.logger.experiment.log({'val_loss': avg_loss})
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        if len(test_label.shape) == 1:
            if self.params['dataset'] == 'wikiart':
                num_classes = 27
            else:
                num_classes = -1
            test_label = torch.nn.functional.one_hot(test_label, num_classes)
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
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

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        try:
            if self.params['name'] is 'CSVAE':
                params_without_delta = [param for name, param in self.model.named_parameters() if
                                        'decoder_z_to_y' not in name]
                params_delta = [param for name, param in self.model.named_parameters() if 'decoder_z_to_y' in name]

                opt_without_delta = optim.Adam(params_without_delta, lr=(1e-3) / 2)
                scheduler_without_delta = optim.lr_scheduler.MultiStepLR(opt_without_delta,
                                                                         milestones=[pow(3, i) for i in range(7)],
                                                                         gamma=pow(0.1, 1 / 7))
                opt_delta = optim.Adam(params_delta, lr=(1e-3) / 2)
                scheduler_delta = optim.lr_scheduler.MultiStepLR(opt_delta, milestones=[pow(3, i) for i in range(7)],
                                                                 gamma=pow(0.1, 1 / 7))

                optims.append(opt_without_delta)
                optims.append(opt_delta)
                scheds.append(scheduler_without_delta)
                scheds.append(scheduler_delta)
                return optims, scheds
        except:
            pass

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            dataset = MNIST(root=self.params['data_path'],
                            train=True,
                            transform=transform,
                            download=True)
        elif self.params['dataset'] == 'cifar10':
            dataset = CIFAR10(root=self.params['data_path'],
                              train=True,
                              transform=transform,
                              download=True)
        elif self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'wikiart':
            main_dataset = datasets.ImageFolder(root=self.params['data_path'],
                                                transform=transform)
            train_size = int(0.8 * len(main_dataset))
            test_size = len(main_dataset) - train_size
            dataset, _ = torch.utils.data.random_split(main_dataset, [train_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=12,
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            self.sample_dataloader = DataLoader(MNIST(root=self.params['data_path'],
                                                      train=True,
                                                      transform=transform,
                                                      download=True),
                                                batch_size=144,
                                                num_workers=12,
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'cifar10':
            self.sample_dataloader = DataLoader(CIFAR10(root=self.params['data_path'],
                                                        train=False,
                                                        transform=transform,
                                                        download=True),
                                                batch_size=144,
                                                num_workers=12,
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(CelebA(root=self.params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                num_workers=12,
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'wikiart':
            main_dataset = datasets.ImageFolder(root=self.params['data_path'] + 'wikiart',
                                                transform=transform)
            train_size = int(0.8 * len(main_dataset))
            test_size = len(main_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(main_dataset, [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
            self.sample_dataloader = DataLoader(test_dataset,
                                                num_workers=12,
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.params['dataset'] == 'cifar10':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'wikiart':
            transform = transforms.Compose([transforms.Resize(64),
                                            transforms.RandomCrop(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        else:
            raise ValueError('Undefined dataset type')
        return transform
