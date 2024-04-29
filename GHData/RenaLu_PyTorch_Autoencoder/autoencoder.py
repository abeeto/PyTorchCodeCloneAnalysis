from collections import OrderedDict

from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

import torch
import torch.nn as nn
from torch import optim
import string
import random
import torch.utils.data as Data


class Encoder(nn.Module):
    def __init__(self, sizes):
        super(Encoder, self).__init__()
                
        layers_en = OrderedDict()       
        for i in range(len(sizes)-1):
            layer_name = 'linear{}'.format(i+1)
            act_name = 'activation{}'.format(i+1)
            layers_en[layer_name] = nn.Linear(sizes[i], sizes[i+1])
            if i==0:
                nn.init.xavier_uniform_(layers_en[layer_name].weight)
            layers_en[act_name] = nn.ReLU()
        
        self.encoder = nn.Sequential(layers_en)

    def forward(self, x):
        return self.encoder(x) 
    
class Decoder(nn.Module):
    def __init__(self, sizes):
        super(Decoder, self).__init__()
        
        sizes = sizes[::-1]
        
        layers_de = OrderedDict()
        for i in range(len(sizes)-2):
            layer_name = 'linear{}'.format(i+1)
            act_name = 'activation{}'.format(i+1)
            layers_de[layer_name] = nn.Linear(sizes[i], sizes[i+1])
            layers_de[act_name] = nn.ReLU()

        layers_de['linear{}'.format(len(sizes)-1)] = nn.Linear(sizes[-2], sizes[-1])
        layers_de['sigmoid'] = nn.Sigmoid()
        self.decoder = nn.Sequential(layers_de)

    def forward(self, encoded):
        return self.decoder(encoded)

# Can be customized according to need (i.e. combining loss functions or change weights
def loss_func(data, decoded):
    cossim_loss = nn.CosineEmbeddingLoss() # Pytorch built-in Cosine similarity for calculating loss 
    y = torch.tensor(np.ones((data.shape[0], 1)), dtype=torch.float).cuda()
    mse_loss = nn.MSELoss()
    loss = cossim_loss(data, decoded, y)
            
    return loss

def training(encoder, decoder, batch_size):
  optimizer_en = optim.Adam(encoder.parameters(), lr=lr)
  scheduler_en = optim.lr_scheduler.ReduceLROnPlateau(optimizer_en, 'min', patience=patience, min_lr=min_lr, factor=0.1)
  optimizer_de = optim.Adam(decoder.parameters(), lr=lr)
  scheduler_de = optim.lr_scheduler.ReduceLROnPlateau(optimizer_de, 'min', patience=patience, min_lr=min_lr, factor=0.1)

  def process_function(engine, batch):
    encoder.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_de.zero_grad()
    encoded = encoder(batch)
    decoded = decoder(encoded)
    loss = criterion(decoded, batch)
    loss.backward()

    optimizer_en.step()
    optimizer_de.step()
    return loss.item()
  

  def eval_function(engine, batch):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoded = encoder(batch)
        decoded = decoder(encoded)
        return decoded, batch
  
  trainer = Engine(process_function)
  train_evaluator = Engine(eval_function)
  validation_evaluator = Engine(eval_function)

  metric = Loss(criterion)
  metric.attach(train_evaluator, 'loss')
  metric.attach(validation_evaluator, 'loss')

  pbar = ProgressBar(persist=True, bar_format="")
  pbar.attach(trainer, ['loss'])

  @trainer.on(Events.EPOCH_COMPLETED)

  def log_training_results(engine):
      train_evaluator.run(train_iterator)
      metrics = train_evaluator.state.metrics
      avg_loss = metrics['loss']    
      
      pbar.log_message(
          "Training Results - Epoch: {}  Avg loss: {:.4f}"
          .format(engine.state.epoch, avg_loss))
      
  def log_validation_results(engine):
      validation_evaluator.run(valid_iterator)
      metrics = validation_evaluator.state.metrics
      avg_loss = metrics['loss']
      print(avg_loss)
      print("Current lr: {}".format(optimizer_de.param_groups[0]['lr']))
      scheduler_en.step(avg_loss)
      scheduler_de.step(avg_loss)
      pbar.log_message(
          "Validation Results - Epoch: {}  Avg loss: {:.4f}"
          .format(engine.state.epoch, avg_loss))
      pbar.n = pbar.last_print_n = 0

  trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

  # Reduce on Plateau
  def average_loss(engine):
    print("Current lr: {}".format(optimizer_de.param_groups[0]['lr']))
    average_loss = engine.state.metrics['loss']
    scheduler_en.step(average_loss)
    scheduler_de.step(average_loss)

  validation_evaluator.add_event_handler(Events.COMPLETED, average_loss)
  
  # Early Stopping
  def score_function(engine):
      val_loss = engine.state.metrics['loss']
      return -val_loss

  handler = EarlyStopping(patience=100, score_function=score_function, trainer=trainer)
  validation_evaluator.add_event_handler(Events.COMPLETED, handler)

  # Model Checkpoint
  checkpointer = ModelCheckpoint(str(DRIVE_PATH.joinpath('models')), 'review', save_interval=10, n_saved=1, create_dir=False, save_as_state_dict=True, require_empty=False)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'encoder': encoder, 'decoder': decoder})

  train_iterator = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
  valid_iterator = Data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

  trainer.run(train_iterator, max_epochs=500)
