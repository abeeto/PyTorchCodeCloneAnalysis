import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from skimage.io import imsave
from torchvision.utils import make_grid
from pathlib import Path
import torch.nn.functional as F
from models import ResNet50
from torch.optim import lr_scheduler
from datetime import datetime
import yaml
from diffusion import LossAwareSampler


class DiffusionWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        diffusion,
        image_size,
        config,
        sampler,
        log_folder='tmp',
        log=True
    ):
        super().__init__()
        self.image_size = image_size
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.min_loss = np.inf
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.sampler = sampler
        self.guiding_cls = ResNet50()
        self.guiding_cls.to(self.device)
        self.guiding_cls.load_state_dict(
            torch.load(self.config['DIFFUSION']['guiding_classifier'], map_location='cpu')
        )
        self.guiding_cls.eval()
        #self.save_hyperparameters()
        self.wandb_log = log
        if self.wandb_log:
            wandb.init(project='PyTorch-Diffusion', config=config)

    def forward(self, x):
        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_sizeim),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            progress=True
        )
        return sampled

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.shape[0]

        # Sample according to sampler
        t, weights = self.sampler.sample(batch_size, self.device)
        #t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()

        loss = self.diffusion.training_losses(
            self.model, imgs, t,
            model_kwargs={'y': labels}
        )

        if isinstance(self.sampler, LossAwareSampler):
            self.sampler.update_with_local_losses(
                t, loss["loss"].detach()
            )

        loss["loss"] = (loss["loss"] * weights)

        return loss

    def training_step_end(self, loss):
        for k in loss.keys():
            loss[k] = loss[k].mean()
        return loss

    def training_epoch_end(self, outputs):
        losses = {}
        for k in outputs[0].keys():
            losses[k] = np.mean([l[k].cpu().item() for l in outputs])
        self.log_dict(losses)
        loss = losses['loss']

        def cond_fn(x, t, y):
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                self.guiding_cls.zero_grad()
                logits = self.guiding_cls(x_in)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return 5 * torch.autograd.grad(selected.sum(), x_in)[0]

        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_size),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            cond_fn=None,
            progress=True
        )
        sampled = (sampled + 1) * 127.5
        sampled = torch.clamp(sampled, 0, 255)
        grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0))).astype(np.uint8)
        imsave(str(self.log_folder / f'test-{self.current_epoch:04d}_regular.png'), grid)

        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_size),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            cond_fn=cond_fn,
            progress=True
        )
        sampled = (sampled + 1) * 127.5
        sampled = torch.clamp(sampled, 0, 255)
        grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0))).astype(np.uint8)
        imsave(str(self.log_folder / f'test-{self.current_epoch:04d}_guided.png'), grid)

        wandb.log(losses)
        wandb.log({'generated_images': [wandb.Image(grid, caption='guided')]}, step=self.current_epoch)

        if self.min_loss > loss:
            print(f'Loss decreased from {self.min_loss} to {loss}.')
            self.min_loss = loss
        self.current_epoch += 1

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config['RUN']['lr']),
            weight_decay=float(self.config['RUN']['weight_decay'])
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=float(self.config['RUN']['lr_gamma']),
            patience=int(self.config['RUN']['lr_decay_step_size'])
        )
        return [optimizer], {
            "scheduler": scheduler,
            "monitor": "loss"
        }
