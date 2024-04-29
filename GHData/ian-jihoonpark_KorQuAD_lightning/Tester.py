import os
import sys
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import KorQuadDataModule
from models import KorQA
import opts
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from utils import evaluate

AVAIL_GPUS = torch.cuda.device_count()

if __name__ == '__main__':
  seed_everything(42)
  args = opts.get_args()

  wandb_logger = WandbLogger(project="KorQuAD", name =args.experiment_name)
  dm = KorQuadDataModule(args)
  dm.setup()

  model = KorQA(hparams =args)

  # Checkpoint call back
  now = datetime.datetime.now()
  nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
 
  # Only use 1 gpu, because of distributed data
  trainer = Trainer(
    max_epochs=args.max_epochs,
    accelerator = "gpu",
    gpus= args.ngpu,
    # Only for test
    limit_train_batches=0, 
    limit_val_batches=0
    )
# Only use 1 gpu, because of distributed data
  trainer.fit(model=model, datamodule= dm)
  trainer.test(
    test_dataloaders=dm.test_dataloader(),
    ckpt_path = args.checkpoints_dir_callback
    )
  # Exact match and F1 score
  expected_version = 'KorQuAD_v1.0'
  with open(args.data_file) as dataset_file:
      dataset_json = json.load(dataset_file)
      read_version = "_".join(dataset_json['version'].split("_")[:-1])
      if (read_version != expected_version):
          print('Evaluation expects ' + expected_version +
                ', but got dataset with ' + read_version,
                file=sys.stderr)
      dataset = dataset_json['data']
  with open(args.prediction_file) as prediction_file:
      predictions = json.load(prediction_file)
  print(json.dumps(evaluate(dataset, predictions)))

