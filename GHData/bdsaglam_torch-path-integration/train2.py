import pathlib
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monty.collections import AttrDict
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.backends import cudnn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch_path_integration import cv_ops, visualization
from torch_path_integration.datasets import WestWorldDataset, MouseDataset
from torch_path_integration.model2 import ContextAwarePathIntegrator
from torch_path_integration.optimizers import RAdam, LookAhead
from torch_path_integration.visualization import PathVisualizer

Tensor = torch.Tensor


class PIMExperiment2(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams
        self.model = ContextAwarePathIntegrator(**hparams.model)
        self.path_vis = self.make_path_visualizer()
        self.mask_cache = []

    def make_path_visualizer(self):
        bg_image = None

        root = pathlib.Path(self.hparams.data_dir)
        fp = root / 'top_view.png'
        if fp.exists():
            from PIL import Image

            image = Image.open(fp)
            img = np.asarray(image)

            pad = 36
            crop_img = img[pad:-pad, pad:-pad]

            h, w, c = crop_img.shape
            alpha = np.full((h, w, 1), fill_value=100)
            bg_image = np.concatenate([crop_img, alpha], -1)

        pv = PathVisualizer(
            rect=(-1, 1, 1, -1),
            figsize_per_example=(6, 6),
            bg_image=bg_image,
            marker_cycle=['.', '^', '.'],
            color_cycle=['blue', 'green', 'red'],
        )
        return pv

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        # optimizer args
        parser.add_argument('--optimizer_type', type=str, default='RMSprop')
        parser.add_argument('--learning_rate', type=float, default=3e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--look_ahead', action='store_true')
        parser.add_argument('--look_ahead_k', type=int, default=5)
        parser.add_argument('--look_ahead_alpha', type=float, default=0.5)
        parser.add_argument('--use_lr_scheduler', action='store_true')
        parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.95)
        return parser

    def forward(self, action_embedding, hidden_state) -> Tensor:
        return self.model(action_embedding, hidden_state)

    def configure_optimizers(self):
        eps = 1e-2 / float(self.hparams.batch_size) ** 2
        if self.hparams.optimizer_type == "RMSprop":
            optimizer = RMSprop(self.parameters(),
                                lr=self.hparams.learning_rate,
                                momentum=0.9,
                                eps=eps,
                                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "RAdam":
            optimizer = RAdam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              eps=eps,
                              weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "Adam":
            optimizer = Adam(self.parameters(),
                             lr=self.hparams.learning_rate,
                             eps=eps,
                             weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")

        if self.hparams.look_ahead:
            optimizer = LookAhead(optimizer,
                                  k=self.hparams.look_ahead_k,
                                  alpha=self.hparams.look_ahead_alpha)

        if not self.hparams.use_lr_scheduler:
            return optimizer

        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.hparams.lr_scheduler_decay_rate)

        return [optimizer], [scheduler]

    def prepare_data(self):
        if self.hparams.dataset_type == 'westworld':
            ds = WestWorldDataset(self.hparams.data_dir)
        elif self.hparams.dataset_type == 'mouse':
            ds = MouseDataset(self.hparams.data_dir, self.hparams.env_size)
        else:
            raise ValueError(f"Unknown dataset type {self.hparams.dataset_type}")

        n = len(ds)
        tn = int(n * 0.8)
        vn = n - tn
        self.train_dataset, self.val_dataset = random_split(ds, [tn, vn])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def get_mask(self, batch, batch_idx):
        if len(self.mask_cache) > batch_idx:
            return self.mask_cache[batch_idx]

        T = batch[0][2].shape[1]
        B = self.hparams.batch_size

        mask = torch.rand(B, T) < self.hparams.anchor_rate  # (B, T)
        self.mask_cache.append(mask)
        return mask

    def inference(self, batch, batch_idx):
        # (B, 1, 2), (B, 1, 1), (B, T, A), (B, T, 2), (B, T, 1)
        (initial_location, initial_orientation, action), (target_location, target_orientation) = batch
        B, T = target_location.shape[:2]

        # (B, 1, 3, 3)
        t_init = cv_ops.affine_transform_2d(
            rotation=initial_orientation,
            trans_x=initial_location[:, :, 0:1],
            trans_y=initial_location[:, :, 1:2],
        )
        # (B, T, 3, 3)
        t_target = cv_ops.affine_transform_2d(
            rotation=target_orientation,
            trans_x=target_location[:, :, 0:1],
            trans_y=target_location[:, :, 1:2],
        )
        mask = self.get_mask(batch, batch_idx)[:B]
        t_out = self.model(t_init, action, t_target, mask)

        return AttrDict(t_out=t_out,
                        mask=mask,
                        target_orientation=target_orientation,
                        target_location=target_location)

    def loss(self, res):
        t_target = cv_ops.make_transform(res.target_orientation, res.target_location)
        return F.mse_loss(res.t_out, t_target, reduction='sum') / t_target.shape[0]

    def training_step(self, batch, batch_idx):
        res = self.inference(batch, batch_idx)
        loss = self.loss(res)

        log = dict(
            loss=loss.detach(),
        )
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        res = self.inference(batch, batch_idx)
        loss = self.loss(res)
        out = dict(val_loss=loss)
        if batch_idx == 0:
            out.update(batch=batch)
            out.update(res=res)
        return out

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_val_loss}

        # log predictions
        batch = outputs[0]['batch']
        res = outputs[0]['res']
        self.visualize_episode(batch, res)

        return {'val_loss': avg_val_loss, 'log': log}

    def visualize_episode(self, batch, res):
        # (B, 1, 2), (B, 1, 1), (B, T, A), (B, T, 2), (B, T, 1)
        (initial_location, initial_orientation, velocity), (target_location, target_orientation) = batch
        B = min(8, target_location.shape[0])
        T = target_location.shape[1]

        rot, sx, sy, sh, tx, ty = cv_ops.decompose_transformation_matrix(res.t_out[:B])
        pred_path = torch.cat([tx, ty], -1).detach().numpy()

        gt_path = torch.cat([initial_location[:B], target_location[:B]], 1).numpy()

        mask_path_list = [target_location[i, res.mask[i, :]].numpy() for i in range(B)]

        loc_fig = self.path_vis.plot(gt_path, mask_path_list, pred_path)
        loc_vis = visualization.fig_to_tensor(loc_fig)
        plt.close(loc_fig)
        self.logger.experiment.add_image('paths', loc_vis, self.current_epoch)


if __name__ == '__main__':
    # For reproducibility
    seed_everything(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    parser = ArgumentParser()
    parser.add_argument('--hparams_file', type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare hparams
    hparams_file = pathlib.Path(args.hparams_file)
    hparams = yaml.safe_load(hparams_file.read_text())

    experiment = PIMExperiment2(
        hparams=Namespace(**hparams),
    )

    # prepare trainer params
    trainer_params = hparams['trainer']
    trainer = Trainer(**trainer_params)
    trainer.fit(experiment)
