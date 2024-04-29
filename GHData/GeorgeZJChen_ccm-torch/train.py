from __future__ import print_function, division
import torch
import numpy as np
import time
import logging
import random

from functions import *
from model import *
from test import *
from data import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('resume', nargs='?', default='0')
args = parser.parse_args()

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

print("Initialising Tensors")
torch.cuda.empty_cache()
net = Net()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = net.to(device)
net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net)


# print('model put to', device)
print('total number of parameters:', total_parameters(net))

if not os.path.exists('./output'):
  os.makedirs('./output')
saver = Saver(model=net, path='./output/model', max_to_keep=4)
best_saver = Saver(model=net, path='./output/best_model', max_to_keep=1)

new_model = args.resume!='1'
batch_size = 8
part = 'B'

logging.basicConfig(filename='./output/train.log',level=logging.INFO)
train_names, test_names = get_data_names(part=part)

print("Training begins")
if new_model:
  set_pretrained(net)
  global_step = 0
  EMA = 0
  train_MAEs = None
  test_MAEs = None
  best_result = 200
else:
  global_step = saver.restore( saver.last_checkpoint() )
  EMA = 0
  train_MAEs = None
  test_MAEs = None
  best_result = float(best_saver.last_checkpoint().split('-')[1])

def learning_rate_scheduler(global_step):
  if global_step < 25000:
    lr = 1e-4
  elif global_step < 50000:
    lr = 5e-5
  elif global_step < 75000:
    lr = 1e-5
  elif global_step < 100000:
    lr = 5e-6
  return lr

try:
  for step in range(global_step, 100000):
    train_inputs, train_targets = next_batch(batch_size, train_names)

    if step%25000==0 and not step==0:
      best_saver.restore( best_saver.last_checkpoint() )

    train_D, train_loss, train_m = net.train(global_step, train_inputs , train_targets, learning_rate_scheduler)
    train_loss = float(to_np(train_loss))

    if EMA == 0:
      EMA = train_loss
    else:
      EMA = moving_average(train_loss, EMA)
    if step%10==0:

      train_t15, train_t14, train_t13, train_t12, train_t11, train_t10 = [np.transpose(t,(0,2,3,1)) for t in train_targets]
      train_out15, train_out14, train_out13, train_out12, train_out11, train_out10 = [ to_np(tensor) for tensor in train_D]

      test_inputs, test_targets = next_batch(batch_size, test_names)

      test_D, test_loss = net.test(test_inputs, test_targets)
      test_loss = float(to_np(test_loss))

      test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = [np.transpose(t,(0,2,3,1)) for t in test_targets]
      test_out15, test_out14, test_out13, test_out12, test_out11, test_out10 = [ to_np(tensor) for tensor in test_D]

      if train_MAEs is None:
        train_MAEs = [ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13)
                      , MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)]
      else:
        train_MAEs = moving_average_array([ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13),
                MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)], train_MAEs)
      if test_MAEs is None:
        test_MAEs = [ MAE(test_out15, test_t15), MAE(test_out14, test_t14), MAE(test_out13, test_t13)
                     , MAE(test_out12, test_t12), MAE(test_out11, test_t11), MAE(test_out10, test_t10)]
      else:
        test_MAEs = moving_average_array([MAE(test_out15,test_t15), MAE(test_out14,test_t14),
                                          MAE(test_out13,test_t13), MAE(test_out12,test_t12),
                MAE(test_out11,test_t11), MAE(test_out10,test_t10)], test_MAEs)

      log_str = ['>>> TRAIN', time.asctime()[10:20]+': i [', str(global_step), '] || [loss, EMA]: [',
                 str(round(train_loss, 2))+', ', str(round(EMA,2)), '] || [EMAoMAE]:', str(train_MAEs),
                 '] || [monitor]:', str(train_m)]
      print(*log_str)
      logging.info(' '.join(log_str))

      log_str = ['>>> TEST ', time.asctime()[10:20]+': i [', str(global_step), '] || [EMAoMAE]:', str(test_MAEs)]
      print(*log_str)
      logging.info(' '.join(log_str))

      if step%200==0 and True:

        display_set_of_imgs([train_out14[0], train_t14[0], train_out13[0], train_t13[0], train_out12[0]
                               , train_t12[0], train_out11[0], train_t11[0], train_out10[0], train_t10[0]
                               , denormalize(train_inputs[0])], rows=3, size=2)
        display_set_of_imgs([test_out14[0], test_t14[0], test_out13[0], test_t13[0], test_out12[0]
                             , test_t12[0], test_out11[0], test_t11[0], test_out10[0], test_t10[0]
                             , denormalize(test_inputs[0])], rows=3, size=2)

      if step%100==0:

        saver.save("model-"+str(global_step), global_step=global_step)
        print(">>> Model saved:", global_step)
        logging.info(">>> Model saved: "+str(global_step))

        if global_step>=200 or step==0:
          test_results = full_test(net, part=part)
          log_str = ['>>> TEST ', time.asctime()+': i [', str(global_step),
                     '] || [Result]:', str([round(result, 2) for result in test_results])]
          if round(test_results[0],2) < best_result:
            best_result = round(test_results[0],2)
            best_saver.save("model-"+str(best_result), global_step=global_step)
            log_str.append(' * BEST *')
          print(*log_str)
          logging.info(' '.join(log_str))

    global_step = global_step + 1
except KeyboardInterrupt:
  print('>>> KeyboardInterrupt. Saving model...')
  saver.save("model-"+str(global_step), global_step=global_step)
  print(">>> Model saved", global_step)
