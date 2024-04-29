from utils.Dataset import Dataset
from models.CycleGAN import CycleGAN
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config.test_config import TestConfig
import os
import numpy as np
from PIL import Image

opt = TestConfig().parse()
model = CycleGAN(opt)
model.load_state_dict(torch.load('log/snapshot/'+opt.name+'_snapshot_'+str(opt.epoch)+'.pkl'))
model.eval()
model.cuda()
dataset = Dataset(opt)
data_loader = DataLoader(dataset, batch_size = 1, shuffle = opt.shuffle, num_workers = 4)
pic_dir = opt.pic_dir

for iteration, input in enumerate(data_loader):
    model.deal_with_input(input)
    model.test()
    g_A = model.generated_A.cpu().numpy()
    g_B = model.generated_B.cpu().numpy()
    c_A = model.cycled_A.cpu().numpy()
    c_B = model.cycled_B.cpu().numpy()
    #g_A = Image.fromarray(((g_A+1.)/2.*255).astype(np.uint8).transpose(1,2,0))
    #g_A.save(os.path.join(pic_dir, 'generated_A_'+str(opt.epoch)+'.png'))
    g_B = Image.fromarray(((g_B+1.)/2.*255).astype(np.uint8).transpose(1,2,0))
    g_B.save(os.path.join(pic_dir, 'generated_B_'+str(iteration+1)+'_'+str(opt.epoch)+'.png'))
    #c_A = Image.fromarray(((c_A+1.)/2.*255).astype(np.uint8).transpose(1,2,0))
    #c_A.save(os.path.join(pic_dir, 'cycled_A_'+str(opt.epoch)+'.png'))
    #c_B = Image.fromarray(((c_B+1.)/2.*255).astype(np.uint8).transpose(1,2,0))
    #c_B.save(os.path.join(pic_dir, 'cycled_B_'+str(opt.epoch)+'.png'))