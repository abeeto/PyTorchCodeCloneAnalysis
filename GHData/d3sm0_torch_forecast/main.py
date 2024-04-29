from argparse import ArgumentParser
from typing import Optional, Dict, Union, Callable, Sequence, Any, Tuple

import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
    _prepare_batch,
    Engine,
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MeanAbsoluteError, RootMeanSquaredError, Metric
from ignite.utils import setup_logger
from torch import nn

import config
import models
from autoregressive import _ar
from dataset import get_data_loaders
from utils.viz_utils import plot_output


# from tqdm import tqdm


def create_ar_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    metrics = metrics or {}

    def _inference(
        engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y, y_pred = _ar((x, y), model)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    #    for name, metric in metrics.items():
    #        metric.attach(evaluator, name)
    #
    return evaluator


def run(args, seed):
    config.make_paths()

    torch.random.manual_seed(seed)
    train_loader, val_loader, shape = get_data_loaders(
        config.Training.batch_size,
        proportion=config.Training.proportion,
        test_batch_size=config.Training.batch_size * 2,
    )
    n, d, t = shape
    model = models.ConvNet(d, seq_len=t)

    writer = tb.SummaryWriter(log_dir=config.TENSORBOARD)

    model.to(config.device)  # Move model before creating optimizer
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=config.device
    )
    trainer.logger = setup_logger("trainer")

    checkpointer = ModelCheckpoint(
        config.MODEL,
        model.__class__.__name__,
        n_saved=2,
        create_dir=True,
        save_as_state_dict=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.Training.save_every),
        checkpointer,
        {"model": model},
    )

    val_metrics = {
        "mse": Loss(criterion),
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
    }

    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=config.device
    )
    evaluator.logger = setup_logger("evaluator")

    ar_evaluator = create_ar_evaluator(model, metrics=val_metrics, device=config.device)
    ar_evaluator.logger = setup_logger("ar")

    @trainer.on(Events.EPOCH_COMPLETED(every=config.Training.save_every))
    def log_ar(engine):
        ar_evaluator.run(val_loader)
        y_pred, y = ar_evaluator.state.output
        fig = plot_output(y, y_pred)
        writer.add_figure("eval/ar", fig, engine.state.epoch)
        plt.close()

    # desc = "ITERATION - loss: {:.2f}"
    # pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED(every=config.Training.log_every))
    def log_training_loss(engine):
        # pbar.desc = desc.format(engine.state.output)
        # pbar.update(log_interval)
        if args.verbose:
            grad_norm = torch.stack([p.grad.norm() for p in model.parameters()]).sum()
            writer.add_scalar("train/grad_norm", grad_norm, engine.state.iteration)
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED(every=config.Training.eval_every))
    def log_training_results(engine):
        # pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, engine.state.epoch)
        # tqdm.write(
        #    f"Training Results - Epoch: {engine.state.epoch}  Avg mse: {evaluator.state.metrics['mse']:.2f}"
        # )

    @trainer.on(Events.EPOCH_COMPLETED(every=config.Training.eval_every))
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, engine.state.epoch)
        # tqdm.write(
        #    f"Validation Results - Epoch: {engine.state.epoch}  Avg mse: {evaluator.state.metrics['mse']:.2f}"
        # )

        # pbar.n = pbar.last_print_n = 0

        y_pred, y = evaluator.state.output

        fig = plot_output(y, y_pred)
        writer.add_figure("eval/preds", fig, engine.state.epoch)
        plt.close()

    # @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    # def log_time(engine):
    #    #tqdm.write(
    #    #    f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds"
    #    #)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        ModelCheckpoint.load_objects({"model": model}, ckpt)

    try:
        trainer.run(train_loader, max_epochs=config.Training.max_epochs)
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    # pbar.close()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=None, help="model path",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="general seed",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="...",
    )

    args = parser.parse_args()

    run(args, seed=args.seed)
