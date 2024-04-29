import time
import os
import torch
from misc import AverageMeter
import scipy.misc
from scipy.misc import imsave
import numpy as np
import cv2
import pdb
from dataset import ImageDataset
from torch.utils.data import DataLoader

from config import Struct, load_config, compose_config_str

config_dict = load_config(file_path='./config_sdfnet.yaml')
configs = Struct(**config_dict)
batch_size = configs.batch_size

resolution = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0,1")

                #if self.configs.use_cuda:
                    #point = point.float().to(device)
                    #point1 = point1.float().to(device)
                    #camera_param = camera_param.float().to(device)
                    #image = image.float().to(device)
                    #sdf = sdf.float().to(device)


class sdfTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, configs):
        self.model = model
        self.train_loader = train_loader #change
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.configs = configs

    def train_epoch(self, epoch_num, f):

        torch.cuda.synchronize()


        batch_time = AverageMeter()
        sdf_loss = AverageMeter()

        self.model.train()
        end = time.time()


        for iter_i, batch_data in enumerate(self.train_loader):
            # t0 = time.time()
            torch.cuda.synchronize()
            #print("A")
            #print("iter:" + str(iter_i))
            proj_point = batch_data['proj_point']
            image = batch_data['image']
            point = batch_data['point']
            sdf = batch_data['sdf']

            if self.configs.use_cuda:
                image = image.float().to(device)
                proj_point = proj_point.float().to(device).squeeze(0)
                point = point.float().to(device).squeeze(0)
                sdf = sdf.float().to(device).squeeze(0)

            # visualize input image
            # img = image[0, :, :, :]
            # img = img.numpy()
            # img = np.transpose(img, (1, 2, 0))
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            # t1 = time.time()
            # print("TRAIN_PART 0")
            # print(t1-t0)

            '''
            b = 2048
            #for j in range(configs.batch_size):
            j = 0
            #print("B")
            batch_data1 = point_dataset(path[j])
            #print("C")
            # print("iter:" + str(iter_i))
            point = batch_data1['point'].float().to(device)
            point1 = batch_data1['point1'].float().to(device)
            sdf = batch_data1['sdf'].float().to(device)
            '''

            #t0 = time.time()
            sdf_pred = self.model(point, proj_point, image)
            #t1 = time.time()
            #print("TIME 1")
            #print(t1-t0)


            loss = self.criterion(sdf_pred, sdf)

            ## print(loss)

            # print("loss:" + str(loss.item()))

            sdf_loss.update(loss.data, batch_size)

            self.optimizer.zero_grad()

            #t1 = time.time()
            loss.backward()

            #t2 = time.time()
            #print("TIME 2")
            #print(t2-t1)


            self.optimizer.step()
            # print("loss:" + str(sdf_loss.avg.item()))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            # Add print materials at the end

            if(iter_i%10 == 0):
                print('Epoch: [{0}][{1}/{2}]'.format(epoch_num, iter_i, len(self.train_loader)))
                print("loss:" + str(sdf_loss.avg.item()))
                print("time:" + str(batch_time.avg))
                print("=========================================================")
                #print("sdf_loss:")
                #print(sdf_loss.val.item())
                #print(sdf_loss.avg.item())

                # write loss in log.txt
                # to visualize it by python loss_curve.py
                f.write('Epoch: [{0}][{1}/{2}]'.format(epoch_num, iter_i, len(self.train_loader))+'\n')
                f.write('{}'.format(sdf_loss.avg.item()))
                f.write('\n')
                f.flush()




    def train_epoch1(self, epoch_num, f):
        t0 = time.time()
        torch.cuda.synchronize()


        batch_time = AverageMeter()
        sdf_loss = AverageMeter()

        self.model.train()
        end = time.time()


        for iter_i, batch_data in enumerate(self.train_loader):

            print("iter:" + str(iter_i))
            camera_param = batch_data['camera_param']
            image = batch_data['image']
            path = batch_data['path']

            if self.configs.use_cuda:
                camera_param = camera_param.float().to(device)
                image = image.float().to(device)

            # visualize input image
            # img = image[0, :, :, :]
            # img = img.numpy()
            # img = np.transpose(img, (1, 2, 0))
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            b = 2048
            for j in range(configs.batch_size):

                dataset2 = PointDataset(path[j])
                data_loader2 = DataLoader(dataset2, batch_size=b, shuffle=True)

                for iter_i1, batch_data1 in enumerate(data_loader2):

                    # print("iter:" + str(iter_i))
                    point = batch_data1['point'].float().to(device)
                    point1 = batch_data1['point1'].float().to(device)
                    sdf = batch_data1['sdf'].float().to(device)



                    sdf_pred = self.model(point, point1, image[j:j+1, :, :, :], camera_param[j:j+1, :])





            loss = self.criterion(sdf_pred, sdf)

            ## print(loss)

            # print("loss:" + str(loss.item()))

            sdf_loss.update(loss.data, batch_size)

            self.optimizer.zero_grad()

            loss.backward()


            self.optimizer.step()
            print("loss:" + str(sdf_loss.avg.item()))


            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            # Add print materials at the end

            if(iter_i%1 == 0):
                #t0 = time.time()
                #print('Epoch: [{0}][{1}/{2}]'.format(epoch_num, iter_i, len(self.val_loader)))

                #print("sdf_loss:")
                #print(sdf_loss.val.item())
                #print(sdf_loss.avg.item())

                # write loss in log.txt
                # to visualize it by python loss_curve.py
                f.write('Epoch: [{0}][{1}/{2}]'.format(epoch_num, iter_i, len(self.train_loader))+'\n')
                f.write('{}'.format(sdf_loss.avg.item()))
                f.write('\n')
                f.flush()
