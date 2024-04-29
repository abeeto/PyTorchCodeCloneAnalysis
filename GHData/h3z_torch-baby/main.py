import os

import wandb

import utils
from callback import cache_checkpoints, early_stopping, wandb_callback
from config import config, prep_env
from data import data_loader, data_process, data_reader, data_split
from model import models
from train import losses, optimizers, schedulers, train

utils.fix_random()


def main():
    prep_env()

    # read csv
    df = data_reader.DataReader().train

    # split
    train_df, val_df, test_df = data_split.split(df)

    # preprocess
    processor = data_process.DataProcess(train_df)
    train_df = processor.preprocess(train_df)
    val_df = processor.preprocess(val_df)
    test_df = processor.preprocess(test_df)

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df, is_train=True).get()
    val_ds = data_loader.DataLoader(val_df).get()
    test_ds = data_loader.DataLoader(test_df).get()

    model = models.get()

    # train
    criterion = losses.get()
    optimizer = optimizers.get(model)
    scheduler = schedulers.get(optimizer, train_ds)
    callbacks = [
        cache_checkpoints.CacheCheckpoints(),
        early_stopping.EarlyStopping(),
        wandb_callback.WandbCallback(),
    ]

    for epoch in range(config["~epochs"]):
        loss = train.epoch_train(
            model, optimizer, scheduler, train_ds, criterion, callbacks
        )
        val_loss = train.epoch_val(model, val_ds, criterion, callbacks)
        print(epoch, ": train_loss", loss, "val_loss", val_loss)

        res = [c.on_epoch_end(loss, val_loss, model) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model) for c in callbacks]

    # predict
    preds, gts = train.predict(model, test_ds)

    # post process
    preds, gts = processor.postprocess(preds), processor.postprocess(gts)

    wandb.finish()


if __name__ == "__main__":
    main()
