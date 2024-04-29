#
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#
import os
import shutil
from argparse import ArgumentParser

# 3rd party library
# Reference : https://github.com/lucidrains/ema-pytorch
from ema_pytorch import EMA

# Reference: https://github.com/sovrasov/flops-counter.pytorch.git
from ptflops import get_model_complexity_info

#
import models
from data import create_train_valiad_loader
from engine import Trainer
from optim import GradualWarmupScheduler, SAM
from utils import logger, load_checkpoint, Mix
# from utils import ddp_utils as ddp

def main_train(opts):

    # 
    num_gpus = torch.cuda.device_count()
    setattr(opts, "ddp.num_gpus", num_gpus)
    if num_gpus > 1:
        #
        dist.init_process_group(backend="nccl")

    is_master_node = False
    if dist.get_rank() == 0:
        is_master_node = True

    ### Create model (2 version, naive & EMA) ###
    model_type = getattr(opts, "model.model_type", "swin_transformer")
    input_resolution = getattr(opts, "model.input_resolution", 224)
    model = getattr(models, model_type)(opts)
    model_ema = EMA(
        model,
        beta=getattr(
            opts, "model.ema_momentum", 0.995
        ),  # exponential moving average factor
        update_after_step=100,  # only after this number of .update() calls will it start updating
        update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    ### AMP ###
    # Enable model to use TF32 as storage unit to enhance the training speed
    gradient_scaler = torch.cuda.amp.GradScaler()

    #
    # logger.info(summary(model, torch.randn(1, 3, input_resolution, input_resolution)))
    if torch.cuda.is_available() and is_master_node:
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model,
                (3, input_resolution, input_resolution),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=True,
            )
            logger.info(f"{model_type}")
            logger.info("{:<30}  {:<8}".format("Computational complexity: ", macs))
            logger.info("{:<30}  {:<8}".format("Number of parameters: ", params))

    ### Create tran/valid dataset & dataloader ###
    (
        train_dataset,
        valid_dataset,
        train_loader,
        valid_loader,
    ) = create_train_valiad_loader(opts)

    ### Create loss function ###
    # Create cross entropy loss, and set label smoothing factor (default 0.1)
    label_smoothing = getattr(opts, "optimizer.label_smoothing", 0.1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    ### Create optimizer, learning scheduler ###
    # Optimizer
    lr = getattr(opts, "optimizer.learning_rate", 1e-5)
    weight_decay = getattr(opts, "optimizer.weight_decay", 1e-8)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # TODO : Let user choose whether to use SAM as optimizer
    optimizer_base = torch.optim.SGD
    optimizer = SAM(model.parameters(), optimizer_base, lr=lr, momentum=0.9)

    # Scheduler
    max_epoch = getattr(opts, "scheduler.max_epoch", 100)
    warmup_epoch = getattr(opts, "scheduler.warmup_epoch", 5)
    cosine_tmax_epoch = getattr(opts, "scheduler.cosine_tmax_epoch", 50)
    cosine_eta_min = getattr(opts, "scheduler.min_learning_rate", 1e-8)
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cosine_tmax_epoch, eta_min=cosine_eta_min
    )
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=1,
        total_epoch=warmup_epoch,
        after_scheduler=scheduler_cosine,
    )

    ### Mix-based augmentation ###
    #
    mix = Mix(opts)

    ### Checkpoints ###
    # Check whether there exist checkpoint, if, restore from previous checkpoint
    save_dir = getattr(opts, "save_dir", None)
    device_type = getattr(opts, "model.device_type", "cpu")
    if not torch.cuda.is_available() and device_type == "cuda" and is_master_node:
        logger.info(f"[WARNING] Attribute device_type is set to 'cuda', but platform did not detect cuda. Setting device to 'cpu'.")
        device_type = "cpu"
    assert save_dir is not None, "[ERROR] Attribute save_dir should not be None"

    # First check whether the existing save_dir have any useful information,
    #       if not, then we should remove the directory and create a new one
    if os.path.isdir(save_dir):
        #
        is_anyCheckpoint = False
        for filename in os.listdir(save_dir):
            if filename.endswith(".pt"):
                is_anyCheckpoint = True
                break

        # if confirm that this folder do not have any checkpoint,
        #   then simply remove it, preventing from thorwing error message in load_checkpoint
        if not is_anyCheckpoint:
            shutil.rmtree(save_dir)
            if is_master_node:
                logger.info(f"Delete folder {save_dir} since it didn't content any checkpoint")

    # Load from previous checkpint if there exists a useful checkpoint,
    #   or create a new checkpoint to store the training information
    start_epoch = 0
    metrics = {}
    if os.path.isdir(save_dir):
        # Resume the training that stop from previous training
        (
            model,
            model_ema,
            optimizer,
            gradient_scaler,
            start_epoch,
            metrics,
        ) = load_checkpoint(
            model, model_ema, optimizer, gradient_scaler, save_dir, device_type
        )
    else:
        # Do not have any checkpoint before, creata a new one
        os.makedirs(save_dir)

        # Copy the config file into save_dir folder
        src = getattr(opts, "config", None)
        fileName = src.split("/")[-1]
        dst = f"{save_dir}/{fileName}"
        shutil.copyfile(src, dst)

    #
    
    ### DistributedDataParallel Initialization ###
    num_gpus = torch.cuda.device_count()
    setattr(opts, "ddp.num_gpus", num_gpus)

    if num_gpus == 1:
        #
        if is_master_node:
            logger.info(f"Using Single GPU")
        model = model.to(device_type)
    elif num_gpus > 1:
        #
        if is_master_node:
            logger.info(f"Using Multiple GPUs ({num_gpus})")
        #
        local_rank = getattr(opts, "local_rank", -1)
        device = torch.device(device_type, local_rank)
        model = model.to(device)
        torch.cuda.set_device(local_rank)
        # dist.init_process_group(backend="nccl")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ### Trainer ###
    # Create a Trainer instance and start training
    trainer = Trainer(
        opts=opts,
        model=model,
        model_ema=model_ema,
        train_loader=train_loader,
        valid_loader=valid_loader,
        mix=mix,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_scaler=gradient_scaler,
        save_dir=save_dir,
        start_epoch=start_epoch,
        max_epoch=max_epoch,
        device_type=device_type,
        **metrics,
    )
    trainer.run()


#
if __name__ == "__main__":
    pass
