from utils.Dataset import Dataset
from models.CycleGAN import CycleGAN
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config.train_config import TrainConfig
import os

opt = TrainConfig().parse()
model = CycleGAN(opt)
model.train()
model.cuda()
dataset = Dataset(opt)
data_loader = DataLoader(dataset, batch_size = 1, shuffle = opt.shuffle, num_workers = 4)

for epoch in range(1, 1 + opt.epoches):
    loss_D_A = 0.
    loss_D_B = 0.
    loss_G = 0.
    cyclic_loss = 0.
    
    for iteration, input in enumerate(data_loader):
        model.deal_with_input(input)
        model.optimize()
        loss_D_A += model.loss_D_A
        loss_D_B += model.loss_D_B
        loss_G += model.loss_G
        cyclic_loss += model.cycleloss_AA + model.cycleloss_BB
        if (iteration+1) % 100 == 0:
            iteration_ = iteration + 1
            print('Epoch %d/%d, iteration %d done!' %(epoch, opt.epoches, iteration_))
            print('Cyclic loss %.4f, loss_G %.4f, loss_D_A %.4f, loss_D_B %.4f.'  
                  %(cyclic_loss/iteration_, loss_G/iteration_, loss_D_A/iteration_, loss_D_B/iteration_))
    
    model.adjust_learning_rate()
    
    size = len(data_loader)
    loss_D_A /= size
    loss_D_B /= size
    loss_G /= size
    cyclic_loss /= size
    
    print('Epoch %d/%d, Cyclic loss %.4f, loss_G %.4f, loss_D_A %.4f, loss_D_B %.4f.' 
          %(epoch, opt.epoches, cyclic_loss, loss_G, loss_D_A, loss_D_B))
    log_file = open(os.path.join(opt.log_dir, opt.name + '_' + 'log.txt'), 'a')
    log_file.write('Epoch %d/%d, Cyclic loss %.4f, loss_G %.4f, loss_D_A %.4f, loss_D_B %.4f.\n' 
          %(epoch, opt.epoches, cyclic_loss, loss_G, loss_D_A, loss_D_B))
    log_file.close()
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(opt.snapshot_dir, opt.name + '_' + 'snapshot_' + str(epoch + 1) + '.pkl'))