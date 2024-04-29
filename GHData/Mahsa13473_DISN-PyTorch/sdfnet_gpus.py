import torch
import torch.nn as nn
import time

import pdb

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
#from torchsummary import summary
import torch.nn.functional as F
from projection import project2d

'''
from config import Struct, load_config, compose_config_str

config_dict = load_config(file_path='./config_sdfnet.yaml')
configs = Struct(**config_dict)
batch_size = configs.batch_size
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")

class sdfnet(nn.Module):
    def __init__(self, configs):
        super(sdfnet, self).__init__()

        self.fc = nn.Linear(7*7*512, 1024)
        # encoder = models.vgg16(pretrained = False)
        # self.global_features = nn.Sequential(*list(encoder.children())[:-2])

        # TODO fix pretrained = True -> DONE!
        # VGG16
        vgg_model = models.vgg16(pretrained = True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9])
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])
        self.Conv6 = nn.Sequential(*list(vgg_model.features.children())[30:32])


        # point feature branch
        self.layer1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size = 1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(64, 256, kernel_size = 1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(256, 512, kernel_size = 1)) #, nn.ReLU())

        self.point_features =nn.Sequential(self.layer1, self.layer2, self.layer3)

        # global stream branch
        self.global_stream_layer1 = nn.Sequential(nn.Conv1d(1536, 512, kernel_size = 1), nn.ReLU())
        self.global_stream_layer2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size = 1), nn.ReLU())
        self.global_stream_layer3 = nn.Sequential(nn.Conv1d(256, 1, kernel_size = 1))

        self.global_stream = nn.Sequential(self.global_stream_layer1, self.global_stream_layer2, self.global_stream_layer3)


        # local stream branch
        self.local_stream_layer1 = nn.Sequential(nn.Conv1d(1984, 512, kernel_size = 1), nn.ReLU())
        self.local_stream_layer2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size = 1), nn.ReLU())
        self.local_stream_layer3 = nn.Sequential(nn.Conv1d(256, 1, kernel_size = 1))

        self.local_stream = nn.Sequential(self.local_stream_layer1, self.local_stream_layer2, self.local_stream_layer3)

        # reshape feature maps to the original image size with bilinear interpolation
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up_sample3 = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.up_sample4 = nn.UpsamplingBilinear2d(scale_factor = 8)
        self.up_sample5 = nn.UpsamplingBilinear2d(scale_factor = 16)

        # self.local_empty = torch.empty(batch_size, 1472).float().to(device)


    def forward(self, point, project_point, img):

        batch = point.size(0)
        batch_size = point.size(1) #2048
        #print(batch_size)

        # point feature

        point_feat = self.point_features(point.view(batch*batch_size, 3, 1)).view(batch, batch_size, 512, 1) #(8, 2048, 512, 1)
        #point_feat = torch.rand(batch, 2048, 512, 1).to(device)
        #print("HYY")
        #print(point_feat.shape)


        # ----------------------------------------------------------------------

        # global feature
        # global_feat = self.global_features(img)
        out1 = self.Conv1(img)
        out2 = self.Conv2(out1)
        out3 = self.Conv3(out2)
        out4 = self.Conv4(out3)
        out5 = self.Conv5(out4)
        out6 = self.Conv6(out5)



        global_feat = out6.view(-1, 7*7*512)
        global_feat = F.relu(self.fc(global_feat))

        global_feat = torch.unsqueeze(global_feat, 2) #(batch = 1, 1024, 1)
        global_feat = torch.unsqueeze(global_feat, 1)

        #print(global_feat.shape)

        global_feat =  global_feat.repeat(1, batch_size, 1, 1) #(2048, 1024, 1)
        #global_feat = global_feat.view()

        #print(global_feat.shape)


        #global_feat = torch.empty(batch_size, 1536, 1).float().to(device) #(2048, 1536, 1)

        '''
        for i in range(batch_size):
            # print(i)
            global_feat[i, :, :] =  torch.cat((global_feat1[:, :, :], point_feat[i:i+1, :, :]), dim=1)
        '''


        global_feat = torch.cat((global_feat, point_feat), dim=2)
        #print(global_feat.shape)
        #print("CHECK")


        global_feat = self.global_stream(global_feat.view(batch*batch_size, 1536 , 1)).view(batch, batch_size, 1, 1) # (2048, 1, 1)
        #print(global_feat.shape)




        # ----------------------------------------------------------------------
        # print("local_feature")
        # local feature


        # project_point = project_point.int64() #(2048, 2, 1)
        project_point = project_point.type(torch.int64) #(8, 2048, 2, 1)

        #project_point = torch.ones(batch_size, 2, 1).int().to(device)

        # print(project_point.shape) #(8, 2048, 2, 1)


        # if configs.use_cuda:
        #     local_feat = local_feat.float().to(device)

        #upsampling to create output with the same size
        out2 = self.up_sample2(out2)
        out3 = self.up_sample3(out3)
        out4 = self.up_sample4(out4)
        out5 = self.up_sample5(out5)

        concat_features = torch.cat([out1, out2, out3, out4, out5], 1) #(1, 1472, 224, 224)

        # local_feat = self.local_empty
        local_feat = torch.empty(batch, 1472, batch_size).float().to(device) #(8, 2048, 1472)
        # print(local_feat.shape)

        p = project_point[:, :, 0, 0] #(8, 2048)
        q = project_point[:, :, 1, 0] #(8, 2048)

        # print(p.shape)
        #print(p)

        for i in range(batch):
            #print(concat_features[i, :, p[i, :], q[i, :]].shape)
            local_feat[i, :, :] = concat_features[i:i+1, :, p[i, :], q[i, :]]
        #print("me")
        # print(local_feat.shape)

        local_feat = local_feat.transpose(1, 2) #(2048, 1472)

        local_feat = local_feat.unsqueeze(3)




        # print(local_feat.shape)
        # print(point_feat.shape)

        local_feat = torch.cat((local_feat, point_feat), dim=2) #(2048, 1984, 1)

        local_feat = self.local_stream(local_feat.view(batch*batch_size, 1984 , 1)).view(batch, batch_size, 1, 1) # (2048, 1, 1)
        #local_feat = self.local_stream(local_feat) #(2048, 1, 1)


        sdf = global_feat[:, :, 0, 0] + local_feat[:, :, 0, 0]
        sdf = torch.unsqueeze(sdf, 2)
        #print(sdf.shape)


        # return sdf, point_feat, global_feat, local_feat
        return sdf



if __name__ == '__main__':
    model = sdfnet(0)
    model.float()
    model.to(device)

    batch_size = 8 # do it for every image once!
    image = torch.randn(batch_size, 3, 224, 224)
    point = torch.randn(batch_size, 3, 1)
    proj_point = torch.randn(batch_size, 2, 1)
    #point = torch.FloatTensor([[[-0.4230765],[-0.0604395],[-0.080586]],[[-0.4230765],[-0.0604395],[-0.080586]]])
    print(point.shape)
    #print (mahsa)

    # camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.], [137.30681486, 28.90141833, 0., 0.73950087, 25.]])
    #camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.]])

    batch = 2048
    #point = torch.randn(batch, 3, 1)

    point = point.float().to(device)
    # point1 = point1.float().to(device)
    proj_point = proj_point.float().to(device)
    image = image.float().to(device)
    # sdf = sdf.float().to(device)



    print("Hey")
    print(point.shape)
    print(proj_point.shape)
    print(image.shape)

    print("Network")
    p = model(point, proj_point, image)
    #print(p.size()[0])
    # print(p)
    print(p.shape)


'''
    model = sdfnet(0)
    model.float()
    model.to(device)

    batch_size = 1 # do it for every image once!
    image = torch.randn(batch_size, 3, 224, 224)
    point = torch.randn(batch_size, 3, 1)
    proj_point = torch.randn(batch_size, 2, 1)
    #point = torch.FloatTensor([[[-0.4230765],[-0.0604395],[-0.080586]],[[-0.4230765],[-0.0604395],[-0.080586]]])
    print(point.shape)

    # camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.], [137.30681486, 28.90141833, 0., 0.73950087, 25.]])
    #camera_param = torch.FloatTensor([[293.5551607, 26.85558558, 0., 0.8375818, 25.]])

    batch = 2048
    #point = torch.randn(batch, 3, 1)

    point = point.float().to(device)
    # point1 = point1.float().to(device)
    proj_point = proj_point.float().to(device)
    image = image.float().to(device)
    # sdf = sdf.float().to(device)



    print("Hey")
    print(point.shape)
    print(proj_point.shape)
    print(image.shape)

    print("Network")
    p = model(point, proj_point, image)
    #print(p.size()[0])
    # print(p)
    print(p.shape)



    #for param in model.parameters.project3to2():
    #    param.requires_grad = False

    #for param in model.parameters():
    #    print(param)

    #print(p.shape)
    #print(g.shape)
    #print(l.shape)
'''
