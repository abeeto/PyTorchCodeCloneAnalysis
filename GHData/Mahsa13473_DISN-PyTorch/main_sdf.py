# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.misc import imsave, imread
import numpy as np
import pdb
import time

import cv2

from dataset_sdf import ImageDataset

from sdfnet_sdf import sdfnet
from trainer_sdf import sdfTrainer
from custom_loss import sdf_loss

from inference_marching_cubes import infer
#import pptk

#import open3d as o3d


from misc import save_checkpoint, count_parameters, transfer_optimizer_to_gpu, make_3d_grid
from config import Struct, load_config, compose_config_str

from projection import project2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1")



def train(configs):
    data_dir = '/local-scratch/mma/DISN/version1'
    train_dataset = ImageDataset(data_dir, phase='val1') #train
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=64)

    val_dataset = ImageDataset(data_dir, phase='val1')  # , augmentation=configs.augmentation # use rotation for augmentation
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=64)


    model = sdfnet(configs=configs)
    model.float() # convert model’s parameter data type to float



    # criterion = nn.BCELoss()
    criterion = sdf_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs.lr), weight_decay=configs.decay_rate)

    #scheduler = StepLR(optimizer, step_size=configs.lr_step, gamma=0.1)

    num_parameters = count_parameters(model)
    print('total number of trainable parameters is: {}'.format(num_parameters))


    trainer = sdfTrainer(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,
                            optimizer=optimizer, configs=configs)

    start_epoch = 0

    if configs.resume:
        if os.path.isfile(configs.model_path):
            print("=> loading checkpoint '{}'".format(configs.model_path))
            checkpoint = torch.load(configs.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if configs.use_cuda:
                transfer_optimizer_to_gpu(optimizer)
            print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, start_epoch))
        else:
            print('no checkpoint found at {}'.format(configs.model_path))

    '''
    if configs.use_cuda:
        print("CUDA")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model) #enabling data parallelism
    '''

    if configs.use_cuda:
        #print("CUDA")
        model.to(device)
        criterion.to(device)





    f = open(os.path.join(data_dir, 'log.txt'), 'w')


    ## check projection and sampling distribution for only one sample val1.txt, batch_size=1

    # proj = []
    # h = []
    # w = []
    # sdff = []
    #
    # for iter_i, batch_data in enumerate(val_loader):
    #     print("iter:" + str(iter_i))
    #
    #
    #     # input
    #     img = batch_data['image'].float().to(device)
    #     point = batch_data['point'].float().to(device).squeeze(0)
    #     proj_point = batch_data['proj_point'].float().to(device).squeeze(0)
    #     sdf = batch_data['sdf'].float().to(device).squeeze(0)
    #
    #     #print("C")
    #     # print("iter:" + str(iter_i))
    #
    #     # sdff.append(sdf.item())
    #     print(sdf.shape)
    #     sdff = sdf.cpu().data
    #
    #
    #
    #     project_point = proj_point
    #     project_point = project_point.int()
    #
    #     print(project_point.shape)
    #
    #
    #     # w.append(project_point[0][0][0].item())
    #     # h.append(project_point[0][1][0].item())
    #     w = project_point[:, 0, 0].cpu().data
    #     h = project_point[:, 1, 0].cpu().data
    #     w = w.numpy()
    #     h = h.numpy()
    #
    #     w1 = []
    #     h1 = []
    #
    #     for i in range(2048):
    #         print(sdff[i,0])
    #         if sdff[i, 0]<0.1:
    #             w1.append(w[i])
    #             h1.append(h[i])
    #
    #     print(w.shape)
    #     print(h.shape)
    #
    #     w1 = np.asarray(w1)
    #     h1 = np.asarray(h1)
    #
    #     print(w1.shape)
    #     print(h1.shape)
    #
    #
    #     sns.distplot(sdff)
    #     plt.show()
    #
    #     #visualize input image
    #     img = img[0, :, :, :]
    #     img = img.cpu().numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    #     img = img*255
    #     img = img.astype(int)
    #
    #     print(w)
    #     print(h)
    #
    #     img[np.round(h1).astype(int), np.round(w1).astype(int), 1] = 255 #for RGB = Green color
    #     #cv2.imshow('proj_new1', img)
    #     cv2.imwrite('proj_new_1.png', img)
    #
    #     if iter_i == 10:
    #         print(mahsa)





    ckpt_save_path = './checkpoints'
    print("start_ epoch = "+ str(start_epoch))
    print("num_epochs = "+ str(configs.num_epochs))
    for epoch_num in range(start_epoch, start_epoch + configs.num_epochs):
        #scheduler.step()
        print("epoch_num = " + str(epoch_num))

        trainer.train_epoch(epoch_num, f)



        if epoch_num % configs.val_interval == 0:
            save_checkpoint({
                'epoch': epoch_num + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best=False, checkpoint=ckpt_save_path,
                filename='checkpoint_sdf_module_{}.pth.tar'.format(epoch_num))
    f.close()
    #print("Start Prediction!...")
    #predict_sdf(configs)

def predict_sdf(configs):

    data_dir = '/local-scratch/mma/DISN/version1'
    #test_dataset = sdfDataset(data_dir, phase='test')
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_dataset = ImageDataset(data_dir, phase='val1')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = sdfnet(configs=configs)
    model.float() # convert model’s parameter data type to float
    if configs.use_cuda:
        print("CUDA")
        model.to(device)

    # Modle should be loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if configs.model_path:
        checkpoint = torch.load(configs.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch_num = checkpoint['epoch']
        print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, epoch_num))


    # viz_dir = os.path.join(configs.exp_dir, 'visualization_{}'.format('test'))
    viz_dir = configs.output

    print("visualization directory : " + str(viz_dir))

    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    model.eval()
    resolution = 64
    print(resolution)

    # use this to don't inference on repetetive file
    render_path_test = []
    # disable gradient computation, we only feed forward-pass during inference
    with torch.no_grad():
        for iter_i, batch_data in enumerate(test_loader):
            print("iter:" + str(iter_i))


            # proj_point = batch_data['proj_point']
            image = batch_data['image']
            camera_param = batch_data['camera_param']
            b_min = batch_data['b_min']
            b_max = batch_data['b_max']
            sdf_gt = batch_data['sdf']

            print(b_max[0].item())

            point, point1 = make_3d_grid(b_min, b_max, resolution)

            if configs.use_cuda:
                # proj_point = proj_point.to(device)
                image = image.float().to(device)
                camera_param = camera_param.float().to(device)
                point1 = point1.float().to(device)


            project_point = project2d(point1, camera_param)

            proj_point = torch.empty(point.size(0), 2, 1)
            # compute projection
            for j in range(point.size(0)):
                proj_point[j, :, :] = project_point[j]



            # visualize input image
            # img = img.numpy()
            # img = np.transpose(img, (1, 2, 0))
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            print(point.shape)
            print(proj_point.shape)
            print(image.shape)

            sdf_pred = model(point, proj_point, image)
            print("After Network")




            SDF = np.zeros((resolution+1, resolution+1, resolution+1))
            s = 0
            for i in range(resolution+1):
                for j in range(resolution+1):
                    for k in range(resolution+1):
                        #print(s)
                        SDF[i,j,k] = sdf_pred[s, :]
                        s = s+1
                        # s = i*resolution*resolution + j*resolution + k
            # np_path = os.path.join(viz_dir, '{}.npy'.format(id) )

            iter_i1 = iter_i
            np_path = os.path.join('final_output/{}_{}.npy'.format(iter_i1, b_max[0].item()))
            np.save(np_path, SDF)
            img = image[0, :, :, :]
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            #cv2.imshow('image', img)
            #cv2.waitKey(0)

            img = img*255
            img = img.astype(int)
            cv2.imwrite('final_output/{}.png'.format(iter_i1), img)

            infer(b_max[0].item(), b_max[1].item(), b_max[2].item(), b_min[0].item(), b_min[1].item(), b_min[2].item(), resolution, SDF, iter_i1)

            # gridx = (b_max[0].item()-b_min[0].item())/resolution
            # gridy = (b_max[1].item()-b_min[1].item())/resolution
            # gridz = (b_max[2].item()-b_min[2].item())/resolution
            #
            # xyz = []
            # sigma = 0.1
            #
            # path_file = os.path.join('final_output/{}.txt'.format(iter_i1))
            # f = open(path_file, "w")
            #
            #
            # for i in range(resolution+1):
            #     print(i)
            #     for j in range(resolution+1):
            #         for k in range(resolution+1):
            #             x = b_min[0].item()+i*gridx
            #             y = b_min[1].item()+j*gridy
            #             z = b_min[2].item()+k*gridz
            #
            #             dist = SDF[i, j, k]
            #             if dist <= sigma:
            #                 xyz.append([x, y, z])
            #                 f.write('{},{},{},{}'.format(dist, x, y, z))
            #                 f.write("\n")
            #
            #
            # f.close()
            #
            # print(len(xyz))
            #
            #
            #
            #
            # xyz = np.asarray(xyz)
            # v = pptk.viewer(xyz)
            # v.set(point_size = 0.01)
            #
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # o3d.io.write_point_cloud("check.ply", pcd)


            print(iter_i)
            print("FINISH")
            print("___________________________________________________________")

        '''
        for iter_i, batch_data in enumerate(test_loader):
            SDF = np.zeros((resolution+1, resolution+1, resolution+1))
            sdf_pred = torch.empty((resolution+1) * (resolution+1) * (resolution+1), 1).to(device)
            print("iter:" + str(iter_i))


            # proj_point = batch_data['proj_point']
            image = batch_data['image']
            camera_param = batch_data['camera_param']
            b_min = batch_data['b_min']
            b_max = batch_data['b_max']
            sdf_gt = batch_data['sdf']

            print(b_max[0].item())

            point, point1 = make_3d_grid(b_min, b_max, resolution)

            if configs.use_cuda:
                # proj_point = proj_point.to(device)
                image = image.float().to(device)
                camera_param = camera_param.float().to(device)
                point1 = point1.float().to(device)

            project_point = project2d(point1, camera_param)

            proj_point = torch.empty(point.size(0), 2, 1)

            # compute projection
            for j in range(point.size(0)):
                proj_point[j, :, :] = project_point[j]

            #sdf_pred = model(point, proj_point, image)

            print("AFTER")
            print(sdf_pred.shape)
            print(point.shape)
            print(proj_point.shape)

            for k in range(resolution+1):
                print(k)
                print("***")
                print(k*(resolution+1)*(resolution+1), (k+1)*(resolution+1)*(resolution+1))
                a = k*(resolution+1)*(resolution+1)
                b = (k+1)*(resolution+1)*(resolution+1)
                sdf_pred[a:b, :] = model(point[a:b, :, :], proj_point[a:b, :, :], image)
                print("After Network")




            SDF = np.zeros((resolution+1, resolution+1, resolution+1))

            s = 0
            for i in range(resolution+1):
                print(i)
                for j in range(resolution+1):
                    for k in range(resolution+1):
                        #print(s)
                        SDF[i,j,k] = sdf_pred[s, :]
                        s = s+1
                        # s = i*resolution*resolution + j*resolution + k
            # np_path = os.path.join(viz_dir, '{}.npy'.format(id) )

            iter_i1 = iter_i
            np_path = os.path.join('final_output/{}_{}.npy'.format(iter_i1, b_max[0].item()))
            np.save(np_path, SDF)
            #SDF = np.load('final_output/256.npy')
            print("SDF")
            print(SDF.shape)
            img = image[0, :, :, :]
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            #cv2.imshow('image', img)
            #cv2.waitKey(0)

            img = img*255
            img = img.astype(int)
            cv2.imwrite('final_output/{}.png'.format(iter_i1), img)

            resolution = 64

            infer(b_max[0].item(), b_max[1].item(), b_max[2].item(), b_min[0].item(), b_min[1].item(), b_min[2].item(), resolution, SDF, iter_i1)
            print(b_max[0].item(), b_max[1].item(), b_max[2].item(), b_min[0].item(), b_min[1].item(), b_min[2].item(), resolution, SDF.shape)
            print(b_max)
            print(b_min)

            print(iter_i)
            print("FINISH")
            print("___________________________________________________________")
            '''


                #sdf_pred = self.model(point, point1, image, camera_param)


                # pred_sdf = sdf_pred.squeeze(0).cpu().numpy()
                # gt_sdf = sdf.squeeze(0).cpu().numpy()




if __name__ == '__main__':
    config_dict = load_config(file_path='./config_sdfnet.yaml')
    configs = Struct(**config_dict)
    config_str = compose_config_str(configs, keywords=['lr', 'batch_size'])

    exp_dir = os.path.join(configs.exp_base_dir, config_str)
    print(exp_dir)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    configs.exp_dir = exp_dir


    if configs.seed:
        torch.manual_seed(configs.seed)
        if configs.use_cuda:
            torch.cuda.manual_seed(configs.seed)

        np.random.seed(configs.seed)
        print('set random seed to {}'.format(configs.seed))

    print(configs)

    if configs.phase == 'train':
        print('training phase')
        train(configs)

    elif configs.phase == 'predict':
        print('inference phase')
        predict_sdf(configs)
    else:
        raise NotImplementedError
