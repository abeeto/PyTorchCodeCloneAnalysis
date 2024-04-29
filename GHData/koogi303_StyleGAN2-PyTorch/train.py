import os
from posixpath import join

import argparse
import logging
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import random
from datetime import datetime

from torch.utils.data.dataloader import DataLoader
from torch import nn

from utils import mixing_noise, requires_grad
from dataset import Dataset
from PIL import Image
from models.loss import (
    d_logistic_loss,
    g_nonsaturating_loss,
    d_r1_loss,
    g_path_regularize,
)
from models.model import Discriminator, Generator

""" DDP (Distributed data parallel) """
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

""" 로그 설정 """
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def setup(rank, world_size):
    """DDP 디바이스 설정"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)


def cleanup():
    """Kill DDP process group"""
    dist.destroy_process_group()


def gan_trainer(
    train_dataloader,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    epoch,
    mean_path_length,
    device,
    args,
):
    generator.train()
    discriminator.train()

    iteration = epoch * len(train_dataloader)

    """  트레이닝 Epoch 시작 """
    start = datetime.now()

    for i, hr in enumerate(train_dataloader):
        """LR & HR 디바이스 설정"""
        real_img = hr.to(device)

        """============= 식별자 학습 ============="""
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        """추론"""
        noise = mixing_noise(args.batch_size, args.style_dims, args.mixing, device)
        fake_img, _ = generator(noise)

        """ 식별자 통과 후 loss 계산 """
        real_pred = discriminator(real_img)
        fake_pred = discriminator(fake_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        """ 가중치 업데이트 """
        discriminator.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            discriminator_optimizer.step()

        if args.use_wandb and device == 0:
            wandb.log({"d_real_score": real_pred}, step=iteration)
            wandb.log({"d_fake_score": fake_pred}, step=iteration)
            wandb.log({"d_loss": d_loss}, step=iteration)

        """============= 생성자 학습 ============="""
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        """추론"""
        noise = mixing_noise(args.batch_size, args.style_dims, args.mixing, device)

        """ 식별자 통과 후 loss 계산 """
        fake_img, _ = generator(noise)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        """ 가중치 업데이트 """
        generator.zero_grad()
        g_loss.backward()
        generator_optimizer.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch_size // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.style_dims, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            generator_optimizer.step()

        if args.use_wandb and device == 0:
            wandb.log({"g_loss": g_loss}, step=iteration)
            wandb.log({"path_loss": path_loss}, step=iteration)
            wandb.log({"path_lengths": path_lengths.mean()}, step=iteration)
            if iteration % 1000 == 0:
                wandb.log({"Results": [wandb.Image(fake_img, caption="Epoch{}-step{}_Label".format(epoch, iteration))]})
            iteration += 1
        

    if args.distributed and device == 0:
        """Generator 모델 저장"""
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.module.state_dict(),
                    "optimizer_state_dict": generator_optimizer.state_dict(),
                },
                os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
            )

    if not args.distributed:
        """Generator 모델 저장"""
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": generator_optimizer.state_dict(),
                },
                os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
            )

    """Epoch args.save_every번에 args.save_every번 저장"""
    if epoch % args.save_every == 0:
        """Discriminator 모델 저장"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": discriminator_optimizer.state_dict(),
            },
            os.path.join(args.outputs_dir, "d_epoch_{}.pth".format(epoch)),
        )

    if device == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main_worker(gpu, args):
    if args.distributed:
        args.rank = args.nr * args.gpus + gpu
        setup(args.rank, args.world_size)

    """ GPEN 모델 설정 """
    generator = Generator(
        size=args.patch_size,
        style_dim=args.style_dims,
        n_mlp=args.mlp,
        channel_multiplier=args.channel_multiplier,
        narrow=args.narrows,
        isconcat=args.is_concat
    ).to(gpu)

    discriminator = Discriminator(
        args.patch_size, channel_multiplier=args.channel_multiplier, narrow=args.narrows
    ).to(gpu)

    """ regularzation ratio 설정 """
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    """ Optimizer 설정 """
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    """ epoch 및 기타 설정 """
    g_epoch = 0
    d_epoch = 0
    mean_path_length = 0

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_g):
        checkpoint_g = torch.load(args.resume_g)
        generator.load_state_dict(checkpoint_g["model_state_dict"])
        g_epoch = checkpoint_g["epoch"] + 1
        generator_optimizer.load_state_dict(checkpoint_g["optimizer_state_dict"])
    if os.path.exists(args.resume_d):
        """resume discriminator"""
        checkpoint_d = torch.load(args.resume_d)
        discriminator.load_state_dict(checkpoint_d["model_state_dict"])
        discriminator_optimizer.load_state_dict(checkpoint_d["optimizer_state_dict"])

    """ 데이터셋 설정 """
    train_dataset = Dataset(args.train_dir, args.patch_size)
    train_sampler = None

    if args.distributed:
        generator = DDP(generator, device_ids=[gpu])
        """ 데이터셋 & 데이터셋 설정 """
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    if gpu == 0 or not args.distributed:
        """로그 인포 프린트 하기"""
        logger.info(
            f"StyleGAN2 MODEL INFO:\n"
            f"StyleGAN2 TRAINING INFO:\n"
            f"\tTotal Epoch:                   {args.num_epochs}\n"
            f"\tStart generator Epoch:         {g_epoch}\n"
            f"\tStart discrimnator Epoch:      {d_epoch}\n"
            f"\tTrain directory path:          {args.train_dir}\n"
            f"\tOutput weights directory path: {args.outputs_dir}\n"
            f"\tGAN learning rate:             {args.lr}\n"
            f"\tPatch size:                    {args.patch_size}\n"
            f"\tBatch size:                    {args.batch_size}\n"
        )

    """GAN Training"""
    for epoch in range(g_epoch, args.num_epochs):
        gan_trainer(
            train_dataloader=train_dataloader,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            mean_path_length=mean_path_length,
            device=gpu,
            args=args,
        )


# 874 CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --train-dir /dataset/merged_FFHQ/ --outputs-dir weights_GPEN_STYLEGAN_with_g_reg --patch-size 256 --use-wandb --is-concat &
if __name__ == "__main__":
    """로그 설정"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()

    """data args setup"""
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)

    """model args setup"""
    parser.add_argument(
        "--n-sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path-regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path-batch-shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d-reg-every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g-reg-every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--is-concat", action="store_true")
    parser.add_argument("--style-dims", type=int, default=512)
    parser.add_argument("--mlp", type=int, default=8)
    parser.add_argument("--channel-multiplier", type=int, default=1)
    parser.add_argument("--narrows", type=float, default=0.5)

    """Training details args setup"""
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=31)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume-g", type=str, default="generator.pth")
    parser.add_argument("--resume-d", type=str, default="discriminator.pth")
    parser.add_argument("--patch-size", type=int, default=256)

    """ Distributed data parallel setup"""
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument(
        "-g",
        "--gpus",
        default=0,
        type=int,
        help="if DDP, number of gpus per node or if not ddp, gpu number",
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="StyleGAN2")
        wandb.config.update(args)

    """ weight를 저장 할 경로 설정 """
    args.outputs_dir = os.path.join(args.outputs_dir, f"StyleGAN2")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ Seed 설정 """
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.distributed:
        gpus = gpus = torch.cuda.device_count()
        args.world_size = gpus * args.nodes
        mp.spawn(main_worker, nprocs=gpus, args=(args,), join=True)
    else:
        main_worker(args.gpus, args)
