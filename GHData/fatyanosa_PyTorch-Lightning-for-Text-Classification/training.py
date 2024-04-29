import argparse
import pytorch_lightning as pl
from classifier import Transformer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
import os
import time
from datetime import datetime
import torch

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parser.add_argument('--seed', type=int, default=42, help="Training seed.")

    # Callbacks args
    parser.add_argument("--save_top_k", default=1, type=int, help="The best k models according to the quantity monitored will be saved.")
    parser.add_argument("--metric", default="accuracy", type=str, help="Metric. See https://huggingface.co/metrics for more details.", choices=["accuracy", "f1", "matthews_correlation", "precision", "recall"])
    parser.add_argument("--monitor", default="accuracy", type=str, help="Quantity to monitor.")
    parser.add_argument("--metric_mode", default="max", type=str, help="If we want to min/max the monitored quantity.", choices=["auto", "min", "max"])
    parser.add_argument("--patience", default=5, type=int, help=("Number of epochs with no improvement after which training will be stopped."))
    parser.add_argument("--min_epochs", default=1, type=int, help="Limits training to a minimum number of epochs")
    parser.add_argument("--max_epochs", default=20, type=int, help="Limits training to a max number number of epochs")

    # Batching args
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size to be used.")
    parser.add_argument("--accumulate_grad_batches", default=2, type=int, help=("Accumulated gradients runs K small batches of size N before doing a backwards pass."))

    # GPU args
    parser.add_argument("--gpus", type=int, default=1, help="Which GPU")

    parser.add_argument("--val_check_interval", default=1.0, type=float, help=("If you don't want to use the entire dev set (for debugging or if it's huge), set how much of the dev set you want to use with this flag."))

    # Task-related args
    parser.add_argument("--task_name", default="cola", type=str, help="GLUE task to be used.", choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli', 'ax'])
    parser.add_argument("--text_fields", default='sentence', type=str, help="Text fields.")
    parser.add_argument("--class_names", default='0 1', type=str, help="Class names.")

    parser = Transformer.add_model_specific_args(parser)

    return parser.parse_args(args)

def main(args):
    pl.seed_everything(args.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------
    model = Transformer(**vars(args))

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        min_delta=0.0,
        patience=args.patience,
        verbose=True,
        mode=args.metric_mode,
    )

    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------

    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        "experiments/", tb_logger.version
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename= "model-{epoch:02d}-{val_loss:.3f}-{"+args.monitor+":.3f}",
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.monitor,
        period=1,
        mode=args.metric_mode,
        save_weights_only=True
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        gpus=[args.gpus],
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        val_check_interval=args.val_check_interval

    )
    return model, trainer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    then = time.perf_counter()

    args = parse_args()
    model, trainer = main(args)

    # ------------------------
    # START TRAINING
    # ------------------------
    trainer.fit(model, model.data)

    best_score = trainer.checkpoint_callback.best_model_score.item()

    # time
    torch.cuda.synchronize()

    now = time.perf_counter()
    training_time = now - then

    # save the result
    f_results = open(str(args.task_name)+'.csv','a')
    f_results.write(str(args.learning_rate) + ',' +
                    str(args.batch_size) + ',' +
                    str(args.seed) + ',' +
                    str(best_score) + ',' +
                    str(training_time) + ',' +
                    str(args.model_name_or_path) + ',' +
                    str(count_parameters(model)) + ',' +
                    str(trainer.current_epoch) + ',' +
                    str(trainer.checkpoint_callback.best_model_path) + '\n')
    f_results.close()
