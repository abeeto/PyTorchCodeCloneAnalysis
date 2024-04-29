from __future__ import absolute_import
from basic import Basic
from callback import MetricCallback
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torchvision import models
import optuna
import argparse


def objective(trial):
    metrics_callback = MetricCallback()

    trainer = pl.Trainer(
        max_epochs=50,
        num_sanity_val_steps=-1,
        gpus=[1] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
    )
    hparams = {
        'lr': trial.suggest_float("lr", 0.0006, 0.0008), 
        'mask_size_suggestion': trial.suggest_int("mask_size_suggestion", 4, 50)
    }

    model = Basic(hparams, trial)
    trainer.fit(model)


    return metrics_callback.metrics[-1]


if __name__ == "__main__":
    pruner = optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=5, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("All trials:")
    print(study.trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))