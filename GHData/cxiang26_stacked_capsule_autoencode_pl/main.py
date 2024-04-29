import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import torchvision.datasets
import torchvision.transforms
from torch.backends import cudnn

from model.scae import SCAE as scae
from model.scae import part_encoder, part_decoder, obj_encoder, obj_decoder
from model.eval import classification_probe

class SCAE(pl.LightningModule):
    def __init__(self, n_classes: int=10,
                 n_obj_caps: int=10,
                 n_part_caps: int=40,
                 learning_rate: float=0.2,
                 batch_size: int=128,
                 workers: int=0,
                 max_epochs: int=1000,
                 **kwargs):
        super(SCAE, self).__init__()
        self.save_hyperparameters()

        self.part_encoder = part_encoder(num_capsules=self.hparams.n_part_caps)
        self.part_decoder = part_decoder(num_capsules=self.hparams.n_part_caps, target_size=28)
        self.obj_encoder = obj_encoder(n_outputs=self.hparams.n_obj_caps)
        self.obj_decoder = obj_decoder(n_caps=self.hparams.n_obj_caps, n_caps_dims=2, n_votes=self.hparams.n_part_caps)
        self._scaenet = scae(self.part_encoder, self.part_decoder,
                             self.obj_encoder, self.obj_decoder)
        self.prior_accuracy = classification_probe(self.hparams.n_obj_caps, self.hparams.n_classes)
        self.posterior_accuracy = classification_probe(self.hparams.n_obj_caps, self.hparams.n_classes)
        del self.part_decoder, self.part_encoder, self.obj_decoder, self.obj_encoder

        self.example_input_array = torch.rand(2, 1, 28, 28)

    def forward(self, x):
        res = self._scaenet(x)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        res = self(x)
        loss = self.loss(res)
        xe1, posterior_acc = self.posterior_accuracy(res.posterior_mixing_probs.sum(dim=1), y)
        xe2, prior_acc = self.prior_accuracy(res.caps_presence_prob, y)
        result = pl.TrainResult(minimize=(loss[0]+xe1+xe2))
        result.log_dict({'train_tatol_loss':loss[0]+xe1+xe2, 'train_rec_ll': loss[1], 'train_log_prob': loss[2],
                         'train_dynamic_weight_l2': loss[3], 'train_primary_caps_l1': loss[4],
                         'train_posterior_within_sparsity_loss': loss[5], 'train_posterior_between_sparsity_loss': loss[6],
                         'train_prior_within_sparsity_loss': loss[7], 'train_prior_between_sparsity_loss': loss[8],
                         'train_posterior_cls_acc': posterior_acc, 'train_prior_cls_acc': prior_acc,
                         'train_posterior_cls_xe': xe1, 'train_prior_cls_xe': xe2})
        if batch_idx == 0:
            n = min(self.hparams.batch_size, 8)
            recons = [x.cpu()[:n], res.rec.pdf.mode().cpu()[:n]]
            if res.get('bottom_up_rec'):
                recons.append(res.bottom_up_rec.pdf.mode().cpu()[:n])
            if res.get('top_down_rec'):
                recons.append(res.top_down_rec.pdf.mode().cpu()[:n])
            recon = torch.cat(recons, 0)
            rg = torchvision.utils.make_grid(recon,
                                             nrow=n, pad_value=0, padding=1)
            self.logger.experiment.add_image('train_recons', rg, self.current_epoch)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        res = self(x)
        loss = self.loss(res)
        xe1, posterior_acc = self.posterior_accuracy(res.posterior_mixing_probs.sum(dim=1), y)
        xe2, prior_acc = self.prior_accuracy(res.caps_presence_prob, y)

        result = pl.EvalResult(checkpoint_on=(1-posterior_acc))
        result.log_dict({'val_total_loss': loss[0]+xe1+xe2, 'val_rec_ll': loss[1], 'val_log_prob': loss[2],
                         'val_dynamic_weight_l2': loss[3], 'val_primary_caps_l1': loss[4],
                         'val_posterior_within_sparsity_loss': loss[5], 'val_posterior_between_sparsity_loss': loss[6],
                         'val_prior_within_sparsity_loss': loss[7], 'val_prior_between_sparsity_loss': loss[8],
                         'val_posetrior_cls_acc': posterior_acc, 'val_prior_cls_acc': prior_acc,
                         'val_posterior_cls_xe': xe1, 'val_prior_cls_xe': xe2})
        if batch_idx == 0:
            n = min(self.hparams.batch_size, 8)
            self.logger.experiment.add_images('transformed_templates', res.transformed_templates[0], self.current_epoch)
            recons = [x.cpu()[:n], res.rec.pdf.mode().cpu()[:n]]
            if res.get('bottom_up_rec'):
                recons.append(res.bottom_up_rec.pdf.mode().cpu()[:n])
            if res.get('top_down_rec'):
                recons.append(res.top_down_rec.pdf.mode().cpu()[:n])
            recon = torch.cat(recons, 0)
            rg = torchvision.utils.make_grid(recon,
                                             nrow=n, pad_value=0, padding=1)
            self.logger.experiment.add_image('val_recons', rg, self.current_epoch)
        return result

    def loss(self, res):
        n_points = int(res.posterior_mixing_probs.shape[1])
        mass_explained_by_capsule = torch.sum(res.posterior_mixing_probs, 1)/n_points

        # entropy in capsule activations
        within_example = mass_explained_by_capsule / (torch.sum(mass_explained_by_capsule, 1, keepdim=True) + 1e-8)
        posterior_within_sparsity_loss = torch.mean(-torch.sum(within_example * self.safe_log(within_example), dim=-1))

        between_example = torch.sum(mass_explained_by_capsule, 0)
        between_example = between_example / (torch.sum(between_example, 0, keepdim=True) + 1e-8)
        posterior_between_sparsity_loss = torch.mean(-torch.sum(between_example * self.safe_log(between_example), dim=-1))

        # l2 penalty on capsule activations
        batch_size, num_caps = res.caps_presence_prob.size()
        within_example_constant = float(num_caps) / self.hparams.n_classes
        between_example_constant = float(batch_size) / self.hparams.n_classes
        prior_within_sparsity_loss = torch.sum((torch.sum(res.caps_presence_prob, 1) - within_example_constant) ** 2) / batch_size
        prior_between_sparsity_loss = -torch.sum((torch.sum(res.caps_presence_prob, 0) - between_example_constant) ** 2) / num_caps

        total_loss = (- self.hparams.rec_ll_weight * res.rec_ll
                      - self.hparams.caps_l1_weight * res.log_prob
                      + self.hparams.dynamic_l2_weight * res.dynamic_weights_l2
                      + self.hparams.primary_caps_sparsity_weight * res.primary_caps_l1
                      + self.hparams.posterior_within_example_sparsity_weight * posterior_within_sparsity_loss
                      - self.hparams.posterior_between_example_sparsity_weight * posterior_between_sparsity_loss
                      + self.hparams.prior_within_example_sparsity_weight * prior_within_sparsity_loss
                      - self.hparams.prior_between_example_sparsity_weight * prior_between_sparsity_loss
                      )
        return (total_loss, -res.rec_ll,
                -self.hparams.caps_l1_weight * res.log_prob,
                self.hparams.dynamic_l2_weight * res.dynamic_weights_l2,
                self.hparams.primary_caps_sparsity_weight * res.primary_caps_l1,
                self.hparams.posterior_within_example_sparsity_weight * posterior_within_sparsity_loss,
                -self.hparams.posterior_between_example_sparsity_weight * posterior_between_sparsity_loss,
                self.hparams.prior_within_example_sparsity_weight * prior_within_sparsity_loss,
                -self.hparams.prior_between_example_sparsity_weight * prior_between_sparsity_loss)

    def safe_log(self, tensor, eps=1e-16):
        is_zero = torch.le(tensor, eps)
        tensor = torch.where(is_zero, torch.ones_like(tensor), tensor)
        tensor = torch.where(is_zero, torch.zeros_like(tensor), torch.log(tensor))
        return tensor

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate,
                                  momentum=0.9, eps=(1 / (10 * self.hparams.batch_size) ** 2))
        return [optimizer]

    def train_dataloader(self): # Note that the inputs and templates should scale to [0, 1]
        # image_size = (28, 28)
        # model_input_size = (40, 40)
        # padding = tuple((model_input_size[i] - image_size[i]) // 2
        #                 for i in range(len(model_input_size)))
        # translate = tuple(p / o for p, o in zip(padding, model_input_size))
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.Pad(2), torchvision.transforms.RandomCrop(28),
            # torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
            # torchvision.transforms.RandomAffine(degrees=0, translate=translate, fillcolor=0),
            torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=self.hparams.data_dir, train=True, transform=trans, download=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers)
        return train_loader

    def val_dataloader(self):
        trans = torchvision.transforms.Compose([
            # torchvision.transforms.Pad((6,6), fill=0, padding_mode='constant'),
             torchvision.transforms.ToTensor()])
        val_set = torchvision.datasets.MNIST(root=self.hparams.data_dir, train=True, transform=trans, download=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers)
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset', type=str, default='mnist', help='mnist')

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--workers', default=2, type=int)

        # optim
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)

        # loss weights
        parser.add_argument('--rec-ll-weight', type=float, default=0.2)
        parser.add_argument('--caps-l1-weight', type=float, default=1.)
        parser.add_argument('--dynamic-l2-weight', type=float, default=10.)
        parser.add_argument('--primary-caps-sparsity-weight', type=float, default=0.)
        parser.add_argument('--posterior-within-example-sparsity-weight', type=float, default=10.)
        parser.add_argument('--posterior-between-example-sparsity-weight', type=float, default=10.)
        parser.add_argument('--prior-within-example-sparsity-weight', type=float, default=1.)
        parser.add_argument('--prior-between-example-sparsity-weight', type=float, default=1.)

        # Model
        parser.add_argument('--meta_dir', default='model/', type=str, help='path to meta.bin for imagenet')
        return parser

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, len(args.gpus)))
        args.workers = int(args.workers / max(1, len(args.gpus)))
    model = SCAE(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)

    # if args.evaluate:
    #     trainer.test(model)
    # else:
    trainer.fit(model)

def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')
    parent_parser.add_argument('--n-classes', default=10, type=int)
    parent_parser.add_argument('--n-part-caps', default=24, type=int)
    parent_parser.add_argument('--n-obj-caps', default=24, type=int)
    parser = SCAE.add_model_specific_args(parent_parser)
    parser.set_defaults(
        gpus=[0],
        # distributed_backend='ddp',
        profiler=True,
        deterministic=True,
        max_epochs=600,
        log_save_interval=50,
        # gradient_clip_val=.5,
        # precision=32,
        evaluate=True,
        # resume_from_checkpoint='./lightning_logs/version_16/checkpoints/epoch=87.ckpt',
        # checkpoint_callback=checkpoint_callback,
        # limit_val_batches=0.2, # run through only 25% of the validation set each epoch
        log_gpu_memory='all', # log all the GPUs
    )
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()



