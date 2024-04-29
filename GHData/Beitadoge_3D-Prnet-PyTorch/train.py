# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import argparse

from collections import deque

#用于测试nme￼

from benchmark_afw2000 import cal_aflw2000_nme

#用于pytorch加速￼
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Set random seem for reproducibility
manualSeed = 5
INFO("Random Seed", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

FLAGS = {"start_epoch": 0,
         "target_epoch": 300,
         "device": "cuda",
         "mask_path": "./utils/uv_data/uv_weight_mask_gdh.png",
         "lr": 1e-4,
         "batch_size":16, #16
         "save_interval": 1,
         "normalize_mean": [0.485, 0.456, 0.406],
         "normalize_std": [0.229, 0.224, 0.225],
         "model_path": "./results/12_11/result",
         "summary_path": "./results/12_11/prnet_runs",
         "gauss_kernel": "original",
         "summary_step": 0,
         "resume": True}

#创建目录
if not os.path.exists(FLAGS['model_path']):
    os.makedirs(FLAGS['model_path'])
if not os.path.exists(FLAGS['summary_path']):
    os.makedirs(FLAGS['summary_path'])

#动态调整学习率
def adjust_learning_rate(lr , epoch, optimizer):
    if epoch>0 and (epoch+1)%5 ==0 and lr >1e-7:
        lr = lr *0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main(data_dir):
    # 0) Tensoboard Writer.
    writer = SummaryWriter(FLAGS['summary_path'])
    origin_img, uv_map_gt, uv_map_predicted = None, None, None

    # 1) Create Dataset of 300_WLP.
    train_data_dir = ['/home/beitadoge/Github/PRNet_PyTorch/Data/PRNet_PyTorch_Data/300WLP_AFW_HELEN_LFPW',
                      '/home/beitadoge/Github/PRNet_PyTorch/Data/PRNet_PyTorch_Data/300WLP_AFW_HELEN_LFPW_Flip',
                      '/home/beitadoge/Github/PRNet_PyTorch/Data/PRNet_PyTorch_Data/300WLP_IBUG_Src_Flip']
    wlp300 = PRNetDataset(root_dir=train_data_dir,
                          transform=transforms.Compose([
                                                        ToTensor(),
                                                        ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])]))


    # 2) Create DataLoader.
    wlp300_dataloader = DataLoaderX(dataset=wlp300, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=4)

    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    model = ResFCN256()

    #GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda")

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS["lr"], betas=(0.5, 0.999))
    # scheduler_MultiStepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer,[11],gamma=0.5, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,min_lr=1e-6,verbose=False)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler_StepLR = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5, last_epoch=-1)


    #apex混合精度训练
    # from apex import amp
    # model , optimizer = amp.initialize(model,optimizer,opt_level="O1",verbosity=0)

    #Loss
    stat_loss = SSIM(mask_path=FLAGS["mask_path"], gauss=FLAGS["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAGS["mask_path"])

    # Load the pre-trained weight
    if FLAGS['resume'] and os.path.exists(os.path.join(FLAGS['model_path'], "latest.pth")):
        state = torch.load(os.path.join(FLAGS['model_path'], "latest.pth")) #这是个字典,keys: ['prnet', 'Loss', 'start_epoch']
        model.load_state_dict(state['prnet'])
        optimizer.load_state_dict(state['optimizer'])
        # amp.load_state_dict(state['amp'])
        start_epoch = state['start_epoch']
        INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    else:
        start_epoch = 0
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    #Tensorboard
    model_input = torch.rand(FLAGS['batch_size'],3,256,256)
    writer.add_graph = (model , model_input)

    nme_mean = 999

    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        Loss_list, Stat_list = deque(maxlen=len(bar)) , deque(maxlen=len(bar))

        model.train()
        for i, sample in enumerate(bar):
            uv_map, origin = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])

            # Inference.
            uv_map_result = model(origin)

            # Loss & ssim stat.
            logit_loss = loss(uv_map_result, uv_map)
            stat_logit = stat_loss(uv_map_result, uv_map)

            # Record Loss.
            Loss_list.append(logit_loss.item())
            Stat_list.append(stat_logit.item())

            # Update.
            optimizer.zero_grad()
            logit_loss.backward()
            # with amp.scale_loss(logit_loss,optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            bar.set_description(" {} lr {} [Loss(Paper)] {:.5f} [SSIM({})] {:.5f}".format(ep, lr, np.mean(Loss_list), FLAGS["gauss_kernel"], np.mean(Stat_list)))

            # Record Training information in Tensorboard.
            # if origin_img is None and uv_map_gt is None:
            #     origin_img, uv_map_gt = origin, uv_map
            # uv_map_predicted = uv_map_result

            #写入Tensorboard
            # FLAGS["summary_step"] += 1
            # if  FLAGS["summary_step"] % 500 ==0:
            #     writer.add_scalar("Original Loss", Loss_list[-1], FLAGS["summary_step"])
            #     writer.add_scalar("SSIM Loss", Stat_list[-1], FLAGS["summary_step"])

            #     grid_1, grid_2, grid_3 = make_grid(origin_img, normalize=True), make_grid(uv_map_gt), make_grid(uv_map_predicted)

            #     writer.add_image('original', grid_1, FLAGS["summary_step"])
            #     writer.add_image('gt_uv_map', grid_2, FLAGS["summary_step"])
            #     writer.add_image('predicted_uv_map', grid_3, FLAGS["summary_step"])
            #     writer.add_graph(model, uv_map)
        
        #每个epoch过后将Loss写入Tensorboard
        loss_mean = np.mean(Loss_list)
        writer.add_scalar("Original Loss", loss_mean , ep)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("lr",lr, ep)
        # scheduler_StepLR.step()
        scheduler.step(loss_mean)

        del Loss_list
        del Stat_list

        #Test && Cal AFLW2000's NME
        model.eval()
        if ep % FLAGS["save_interval"] == 0:
            with torch.no_grad():
                nme_mean = cal_aflw2000_nme(model,'/home/beitadoge/Data/PRNet_PyTorch_Data/AFLW2000')
                print("NME IS {}".format(nme_mean))

                writer.add_scalar("Aflw2000_nme",nme_mean, ep)


                origin = cv2.imread("./test_data/obama_origin.jpg")
                gt_uv_map = cv2.imread("./test_data/obama_uv_posmap.jpg")
                origin, gt_uv_map = test_data_preprocess(origin), test_data_preprocess(gt_uv_map)
                origin_in = F.normalize(origin, FLAGS["normalize_mean"], FLAGS["normalize_std"], False).unsqueeze_(0)
                pred_uv_map = model(origin_in).detach().cpu()

                save_image([origin.cpu(), gt_uv_map.unsqueeze_(0).cpu(), pred_uv_map],
                           os.path.join(FLAGS['model_path'], str(ep) + '.png'), nrow=1, normalize=True)

            # # Save model
            # state = {
            #     'prnet': model.state_dict(),
            #     'Loss': Loss_list,
            #     'start_epoch': ep,
            # }
            # torch.save(checkpoint, os.path.join(FLAGS['model_path'], 'epoch{}.pth'.format(ep)))
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'amp': amp.state_dict(),
                'start_epoch': ep
            }
            torch.save(checkpoint, os.path.join(FLAGS['model_path'], 'lastest.pth'))


        # adjust_learning_rate(lr , ep, optimizer)
        # scheduler.step(nme_mean)
        


    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.train_dir)
