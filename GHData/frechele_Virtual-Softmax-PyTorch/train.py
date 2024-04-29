import argparse

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

from virtsm.engine.cifar10 import CIFAR10Module
from virtsm.models.convnet import SimpleConvNet, SimpleConvNetVirt
from virtsm.models.metrics import METRICS


ENGINES = {
    'cifar10': CIFAR10Module
}

MODELS = {
    'simple': SimpleConvNet,
    'simple_virt': SimpleConvNetVirt
}


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(ENGINES.keys()))
    parser.add_argument('--loss', type=str, required=True,
                        choices=list(METRICS.keys()))

    parser.add_argument('--batch_size', type=int, default=32, required=False)

    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_argument()

    model_type = MODELS[args.model]
    engine = ENGINES[args.dataset](model_type, batch_size=args.batch_size, loss=args.loss)

    checkpoint_callback = plc.ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = plc.EarlyStopping(
        verbose=True,
        patience=10,
        monitor='val_acc',
        mode='max'
    )

    trainer_args = {
        'callbacks': [checkpoint_callback, early_stopping],
    }

    trainer = pl.Trainer.from_argparse_args(args, **trainer_args)
    trainer.fit(engine)
