
# Pytorch-Lightning
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch


# Internal Libraries
from src import ViTConfigExtended
from src import Backbone 
from src import LitClassifier 
from src import CIFAR10DataModule

import argparse


if __name__ == "__main__":


   parser = argparse.ArgumentParser(description="Distributed Training",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("--batch_size", help="batch size")
   parser.add_argument("--max_epochs", help="number of epochs")
   parser.add_argument("--accelerator", help="type of distribution example: ddp")
   parser.add_argument("--num_nodes", help="number of nodes")
   parser.add_argument("--gpus", help="number of gpus per each node")
   args = parser.parse_args()
   config = vars(args)
   
   # set the paramerts of ViT Network
   configuration = ViTConfigExtended()

   # Load the dataset
   dm = CIFAR10DataModule(batch_size=int(config['batch_size']), image_size=configuration.image_size)
   dm.prepare_data()
   dm.setup('fit')

   # load the backbone 
   backbone = Backbone(model_type='vit', config=configuration)

   # Initializing the Lightning Model 
   model = LitClassifier(backbone)

   if torch.cuda.is_available():
      trainer = pl.Trainer(gpus=int(config['gpus']), max_epochs=int(config['max_epochs']), accelerator=str(config['accelerator']), num_nodes=int(config['num_nodes']))
   else:
      trainer = pl.Trainer( max_epochs=10, accelerator='ddp', num_nodes=1)
   # Fit the Model 
   trainer.fit(model,dm)
