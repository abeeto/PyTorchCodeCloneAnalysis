# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
# from models.base_model import BaseModel
from torchvision_my import models as BaseModel


mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

class img_classifiction():
    def __init__(self,args) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # data_transform = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #     ]
        # )
        # self.data_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #     ]
        # )
      
             
        self.data_transform =transforms.Compose([
                               transforms.Resize([224,224]),
                                # transforms.Resize([256,256]),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # self.model = BaseModel(name=args.model_name, num_classes=args.num_classes).to(self.device)
        if args.model_name=="mobilenetv3":
            self.model = BaseModel.mobilenet_v3_large(num_classes=args.num_classes).to(self.device)
        elif  args.model_name=="resnet18":
            self.model = BaseModel.resnet18(num_classes=args.num_classes).to(self.device)

        checkpoint_my = torch.load(args.model_weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_my)
        # self.model.load_state_dict(checkpoint_my["model"])
        self.model.eval()

    def predict(self,image):
        # plt.imshow(img)
        # img = data_transform(img)
        # [C, H, W] -> [1, C, H, W]
        # img = torch.unsqueeze(img, dim=0)
        # imgd = self.letterbox(self.img, new_shape=(self.imgsz,self.imgsz))[0]
        # r_image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
        if True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgd = np.array(image,dtype = np.float32) / 255.0
            # image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
            imgd = imgd- np.array(mean)
            imgd =imgd/np.array(std)
            imgd = imgd.transpose(2,0,1)
            imgd = np.ascontiguousarray(imgd)
            imgd = torch.from_numpy(imgd).to(self.device)
            imgd= imgd.float()
             #imgd.half() if self.half else imgd.float()  # uint8 to fp16/32
            if imgd.ndimension() == 3:
                imgd = imgd.unsqueeze(0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgd =Image.fromarray(np.uint8(image))
            imgd = self.data_transform(imgd).to(self.device)
            if imgd.ndimension() == 3:
                imgd = imgd.unsqueeze(0)
        # imgd =Image.fromarray(np.uint8(image))
        # imgd = transforms.ToTensor()(imgd).to(self.device)  #归一化
        # imgd =transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(imgd)  #0 - 255  to -1 - 1
        ##########
        # imgd =Image.fromarray(np.uint8(image))
        # imgd= self.data_transform(imgd)
        # imgd = imgd.to(self.device)
        
        with torch.no_grad():
            output = torch.squeeze(self.model(imgd)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        return  predict,predict_cla

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
def main(args,class_indict):
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), f"file {json_path} does not exist."
    # json_file = open(json_path, 'r')
    # self.class_indict = json.load(json_file)
    classifiction_model = img_classifiction(args)
    img_path = args.img_path
    assert os.path.exists(img_path), f"file {img_path} dose not exist."
    img_list = os.listdir(img_path)
    for im in img_list:
        img_file = os.path.join(img_path,im)
        # img = Image.open(img_file)
        # img_ori = np.array(img) # to opencv 
        # img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
        img_ori = cv2.imread(img_file)
        img_ori = cv2.resize(img_ori, (224,224))
        # img_Image = Image.fromarray(np.uint8(img))  #to Image
        predict,predict_cla = classifiction_model.predict(img_ori)

        # print_res = "real: {}  predict: {} \n   prob: {:.3f}".format(args.real_label, class_indict[str(predict_cla)],
        #                                                         predict[predict_cla].numpy())
        print_res = "pre: {} prob: {:.3f}".format(class_indict[predict_cla],predict[predict_cla].numpy())
        # plt.title(print_res)
        # plt.xticks([])
        # plt.yticks([])
        print(print_res)
        # plt.savefig('./data/predict.jpg', bbox_inches='tight', dpi=600, pad_inches=0.0)
        # plt.show()
        cv2.putText(img_ori, "%s"%print_res, (0, 20), 0, 0.5, [100, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("test",img_ori)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break



if __name__ == '__main__':
    class_name = ["clean_egg","dirt_egg","leak_egg"]
    num = len(class_name)
    # model_path =r"Z:\code\egg_products\dirt_egg_recognition\CV\Image_Classification\weights\resnet.pth"
    # model_path =r"Z:\code\egg_products\dirt_egg_recognition\classification_egg\weight\model_23_83.972.pth"
    model_path =r"Z:\code\egg_products\dirt_egg_recognition\classification_egg\weight\mobilenet\5\model_699_99.607.pth"
    img_path =r"Z:\data\egg_products\dirt\single_egg\train\leak_egg"
    # img_path =r"Z:\data\egg_products\dirt\single_egg\train\dirt_egg"
    # img_path =r"Z:\data\egg_products\dirt\single_egg\train\clean_egg"

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=img_path)
    parser.add_argument('--real_label', type=str, default=class_name[0])
    # parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--model_name', type=str, default='mobilenetv3')
    parser.add_argument('--num_classes', type=int, default=num)
    parser.add_argument('--model_weight_path', type=str, default=model_path)

    args = parser.parse_args()
    main(args,class_name)
