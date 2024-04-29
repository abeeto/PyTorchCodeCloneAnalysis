#
import torch

#
import os
import shutil

# 3rd party library
# Reference : https://github.com/lucidrains/ema-pytorch
from ema_pytorch import EMA

# Reference: https://github.com/sovrasov/flops-counter.pytorch.git
from ptflops import get_model_complexity_info

#
import models
from data import create_train_valiad_loader, create_eval_loader
from engine import Evaluator
from utils import logger, load_checkpoint


def main_eval(opts):

    ### Create model (2 version, naive & EMA) ###
    model_type = getattr(opts, "model.model_type", "swin")
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

    #
    # logger.info(summary(model, torch.randn(1, 3, input_resolution, input_resolution)))
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model,
                (3, input_resolution, input_resolution),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=True,
            )
            print(f"{model_type}")
            print("{:<30}  {:<8}".format("Computational complexity: ", macs))
            print("{:<30}  {:<8}".format("Number of parameters: ", params))

    ### Create tran/valid dataset & dataloader ###
    (
        train_dataset,
        _,
        _,
        _,
    ) = create_train_valiad_loader(opts)
    eval_dataset, eval_loader = create_eval_loader(opts)

    ### Checkpoints ###
    # Check whether there exist checkpoint, if, restore from previous checkpoint
    save_dir = getattr(opts, "save_dir", None)
    device_type = getattr(opts, "model.device_type", "cpu")
    if not torch.cuda.is_available() and device_type == "cuda":
        logger.info(
            "[WARNING] Attribute $device_type is set to 'cuda', but platform did not detect cuda. Setting device to 'cpu'."
        )
        device_type = "cpu"
    assert save_dir is not None, "[ERROR] Attribute $save_dir should not be None"

    # Load from previous checkpint if there exists a useful checkpoint,
    #   or create a new checkpoint to store the training information
    start_epoch = 0
    metrics = {}
    if os.path.isdir(save_dir):
        # Resume the training that stop from previous training
        (
            model,
            model_ema,
            _,
            _,
            _,
            _,
        ) = load_checkpoint(model, model_ema, None, None, save_dir, device_type)
    else:
        raise RuntimeError("[ERROR] Did not find any checkpoint under $save_dir folder")

    ### Evaluator ###
    # Create a Evaluator instance and start training
    evaluator = Evaluator(
        opts=opts,
        model=model,
        model_ema=model_ema,
        eval_loader=eval_loader,
        img_names=None,
        label2class=None,
        device_type=device_type
    )
    evaluator.run()


#
if __name__ == "__main__":
    pass
