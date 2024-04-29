import pathlib
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
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

from torch_path_integration.datasets import WestWorldDataset, MouseDataset
from torch_path_integration.ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from torch_path_integration.model import PathIntegrationModule
from torch_path_integration.optimizers import RAdam, LookAhead
from torch_path_integration.visualization import plot_location_predictions, fig_to_tensor

Tensor = torch.Tensor


class PIMExperiment(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.pce = PlaceCellEnsemble(**hparams.place_cell_ensemble)
        self.hdce = HeadDirectionCellEnsemble(**hparams.head_direction_cell_ensemble)
        self.model = PathIntegrationModule(**hparams.model)
        self.hparams = hparams
        self.current_device = None

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

    def inference(self, batch):
        # (B, 1, 2), (B, 1, 1), (B, T, A), (B, T, 2), (B, T, 1)
        (initial_location, initial_orientation, velocity), (target_location, target_orientation) = batch
        B, T = target_location.shape[:2]

        initial_place = self.pce.encode(initial_location.view(B, -1)).view(B, -1)
        initial_hd = self.hdce.encode(initial_orientation.view(B, -1)).view(B, -1)

        target_place = self.pce.encode(target_location.view(B * T, -1)).view(B, T, -1)
        target_hd = self.hdce.encode(target_orientation.view(B * T, -1)).view(B, T, -1)

        grids = []
        places = []
        hds = []
        hx, cx = self.model.initial_hidden_state(initial_place, initial_hd)
        for i in range(T):
            v = velocity[:, i, :]  # (B, A)
            grid, place, hd, (hx, cx) = self.model(v, (hx, cx))
            grids.append(grid)
            places.append(place)
            hds.append(hd)

        grid = torch.stack(grids, 0).permute(1, 0, 2).contiguous()
        place = torch.stack(places, 0).permute(1, 0, 2).contiguous()
        hd = torch.stack(hds, 0).permute(1, 0, 2).contiguous()

        return AttrDict(
            initial_place=initial_place,
            initial_hd=initial_hd,
            target_place=target_place,
            target_hd=target_hd,
            grid=grid,
            place_log_prob=place,
            hd_log_prob=hd,
        )

    def loss(self, res):
        loss_pc = F.kl_div(res.place_log_prob, res.target_place, reduction='batchmean')
        loss_hdc = F.kl_div(res.hd_log_prob, res.target_hd, reduction='batchmean')
        loss_reg = self.hparams.loss['grid_l2_loss_weight'] * self.model.reg_loss()
        loss = loss_pc + loss_hdc + loss_reg
        return AttrDict(loss=loss, loss_pc=loss_pc, loss_hdc=loss_hdc, loss_reg=loss_reg)

    def training_step(self, batch, batch_idx):
        res = self.inference(batch)
        loss_res = self.loss(res)
        log = dict(
            loss=loss_res.loss.detach(),
            loss_pc=loss_res.loss_pc.detach(),
            loss_hdc=loss_res.loss_hdc.detach(),
            loss_reg=loss_res.loss_reg.detach(),
        )
        return dict(loss=loss_res.loss, log=log)

    def validation_step(self, batch, batch_idx):
        res = self.inference(batch)
        loss_res = self.loss(res)
        out = dict(val_loss=loss_res.loss)
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

        place_prob = res.place_log_prob[:B].exp()

        soft_pred_location = self.pce.decode(place_prob.view(B * T, -1), strategy='soft') \
            .view(B, T, -1).detach().numpy()
        loc_fig = plot_location_predictions(initial_location[:B], soft_pred_location, target_location[:B])
        loc_vis = fig_to_tensor(loc_fig)
        plt.close(loc_fig)

        self.logger.experiment.add_image('location_soft', loc_vis, self.current_epoch)

        hard_pred_location = self.pce.decode(place_prob.view(B * T, -1), strategy='hard') \
            .view(B, T, -1).detach().numpy()
        hard_loc_fig = plot_location_predictions(initial_location[:B], hard_pred_location, target_location[:B])
        hard_loc_vis = fig_to_tensor(hard_loc_fig)
        plt.close(hard_loc_fig)
        self.logger.experiment.add_image('location_hard', hard_loc_vis, self.current_epoch)


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

    experiment = PIMExperiment(
        hparams=Namespace(**hparams),
    )

    # prepare trainer params
    trainer_params = hparams['trainer']
    trainer = Trainer(**trainer_params)
    trainer.fit(experiment)
