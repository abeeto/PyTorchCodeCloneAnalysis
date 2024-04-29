import numpy as np
import torch

from src.trainers.dense_trainer import DenseTrainer
from src.utils.args import get_args
from src.utils.datasets import get_data
from src.utils.logger import get_logger
from src.utils.uncertainty import mcdropout_test, uncertainty_test

# Get parameters and setup directories, loggers
args = get_args()
logger = get_logger(__name__)

# Seed seeds for reproducibility
np.random.seed(args.seed)  # cpu vars
torch.manual_seed(args.seed)  # cpu  vars
torch.cuda.manual_seed_all(args.seed)  # gpu vars

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device

trainers = {
  'dense': DenseTrainer
}


def print_config(model):
  count = 1
  for layer in model.modules():
    if hasattr(layer, 'mask'):
      logger.info('Layer {} Needs Drop {} Drop Prob {}'.format(count,
                                                               layer.needs_drop,
                                                               layer.drop_prob))
      count += 1


def set_drop(model, args):
  total_layers = 0
  for layer in model.modules():
    if hasattr(layer, 'mask'):
      total_layers += 1
  print('Total number of layers {}'.format(total_layers))
  if args.layer_no is None:
    if args.manifold_mcdc:
      mu, sigma = (total_layers - 5), 5
      args.layer_no = np.random.normal(mu, sigma)
    else:
      args.layer_no = total_layers + 1

  logger.info('layer no {}'.format(args.layer_no))
  count = 1
  for layer in model.modules():
    if hasattr(layer, 'mask'):
      if count <= args.layer_no:
        layer.needs_drop = False
        layer.drop_prob = 0
      else:
        layer.needs_drop = True
        layer.drop_prob = args.drop_prob
      count += 1
  print_config(model)

  return model


def main():
  trainloader, testloader = get_data(args)
  # get trainer
  trainer = trainers[args.model_type](args)
  trainer.model = set_drop(trainer.model, args)
  if args.mode == 'train':
    logger.info('Starting training')
    model = trainer.train(trainloader, testloader)
    mcdropout_test(model, testloader, args)
    uncertainty_test(trainer.model, testloader, args)
  elif args.mode == 'eval':
    logger.info('Starting evaluation')
    loss, acc = trainer.test(testloader)
    logger.info("Test Loss: {:.4f} Test Accuracy: {:.4f}".format(loss,
                                                                 acc))
    mcdropout_test(trainer.model, testloader, args)
    uncertainty_test(trainer.model, testloader, args)


if __name__ == '__main__':
  main()
