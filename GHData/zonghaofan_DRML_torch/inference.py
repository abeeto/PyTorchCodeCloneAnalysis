from __future__ import print_function

import os
from PIL import Image
from network_fer2013_deep_short import Network_new
import config as cfg
from data_loader_omg import OMG
from data_loader_fer2013_3channels import FER2013
import torch.utils.data as data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
import cv2
import numpy as np
import h5py

def inference_fer2013_batchsize():
    # test samples
    test_data = FER2013(path='./data/fer2013/data/fer2013_new_data.h5',split='Testing')

    test_loader = data.DataLoader(dataset=test_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=False)
    test_batch_nb = len(test_data)
    print(test_batch_nb)

    model = load_checkpoint('./models_fer2013_deep_short_3channels_48/model.pth')
    print('model=',model)
    for batch_index, (img, label) in enumerate(test_loader):
        img = Variable(img)
        label = Variable(label.squeeze())
        if torch.cuda.is_available():
            model=model.cuda()
            print(True)
            img = img.cuda()
            label=label.squeeze().cuda()
        out = model(img)
        pred = F.softmax(out, dim=1)
        print('pred=',pred)
        predicted = torch.argmax(pred, 1)
        print('predicted=',predicted)
        print('label:====',label)
        acc = (predicted == label.squeeze()).sum().float() / len(label)
        print('acc=',acc)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model
def get_clip_img(img):
    img = cv2.resize(img, (50, 50))
    img_lt = img[:-2,:-2,:]
    img_lt_flip=np.fliplr(img_lt)
    # print(img_lt.shape)
    img_rt=img[:-2,2:,:]
    img_rt_flip=np.fliplr(img_rt)
    # print(img_rt.shape)
    img_lb=img[2:,:-2,:]
    img_lb_flip=np.fliplr(img_lb)
    # print(img_lb.shape)
    img_rb=img[2:,2:,:]
    img_rb_flip=np.fliplr(img_rb)
    # print(img_rb.shape)
    img_center=img[1:-1,1:-1,:]
    img_center_flip=np.fliplr(img_center)
    img_all = np.concatenate((img_lt[np.newaxis, ...], img_rt[np.newaxis, ...], img_lb[np.newaxis, ...],
                              img_rb[np.newaxis, ...], img_center[np.newaxis, ...]), axis=0)

    # img_all=np.concatenate((img_lt[np.newaxis,...],img_rt[np.newaxis,...],img_lb[np.newaxis,...],img_rb[np.newaxis,...],img_center[np.newaxis,...],
    #                         img_lt_flip[np.newaxis,...],img_rt_flip[np.newaxis,...],img_lb_flip[np.newaxis,...],
    #                         img_rb_flip[np.newaxis,...],img_center_flip[np.newaxis,...]),axis=0)
    # print(img_all.shape)
    return img_all
def inference_fer2013():
    import h5py
    import numpy as np
    import cv2
    labels_dict = {'0': 'Anger', '1': 'Disgust', '2': 'Fear', '3': 'Happy', '4': 'Sad', '5': 'Surprise', '6': 'Neutral'}
    model = load_checkpoint('./models_fer2013_deep_short_3channels_48/model.pth')
    path='./data/fer2013/data/fer2013_new_data.h5'
    data = h5py.File(path, 'r')
    print(data.keys())
    for i in data.keys():
        print(i)
    Testing_label = data['Testing_label']
    print(Testing_label.shape)
    print(len(Testing_label))
    Testing_pixel = data['Testing_pixel']
    print(Testing_pixel.shape)

    if torch.cuda.is_available():
        model = model.cuda()
    equal_number=0
    for i,label in enumerate(Testing_label):
        img=Testing_pixel[i].reshape(48,48,1).astype(np.float32)
        s=np.concatenate((img,img,img),axis=-1)
        s = cv2.resize(s, (96, 96))
        b1 = cv2.GaussianBlur(s, (3, 3), 0)
        D1 = ((s - b1) + s)
        # img=np.clip(D1, 0, 255)
        img_all=get_clip_img(D1)
        img_all=torch.from_numpy(np.transpose(img_all,(0,3,1,2))/255.)

    #     # img = tv_F.to_tensor(s/255.)
        print('img=',img.shape)
        if torch.cuda.is_available():
            # img = torch.unsqueeze(img, 0).cuda()
            img_all = img_all.cuda()
        out = model(img_all)
        pred = F.softmax(out, dim=1)
        # print('pred=',pred)
        pred=torch.mean(pred,dim=0,keepdim=True)
        # print('pred=', pred)
        predicted = torch.argmax(pred,dim=1)

        print('label=',label)

        pred_np=predicted.cpu().numpy()[0]
        print('predicted.cpu().numpy()[0]',pred_np)
        if pred_np==label:
            equal_number+=1
        print('face_emotion=',labels_dict[str(pred_np)])
    print('acc=',equal_number/len(Testing_label))

def inference_fer2013_3channels():
    import h5py
    import numpy as np
    import cv2
    labels_dict = {'0': 'Anger', '1': 'Disgust', '2': 'Fear', '3': 'Happy', '4': 'Sad', '5': 'Surprise', '6': 'Neutral'}
    model = load_checkpoint('./models_fer2013_deep_short_3channels_48/model.pth')
    path='./data/fer2013/data/fer2013_new_data.h5'
    data = h5py.File(path, 'r')
    print(data.keys())
    for i in data.keys():
        print(i)
    Testing_label = data['Testing_label']
    print(Testing_label.shape)
    print(len(Testing_label))
    Testing_pixel = data['Testing_pixel']
    print(Testing_pixel.shape)

    if torch.cuda.is_available():
        model = model.cuda()
    equal_number=0
    for i,label in enumerate(Testing_label):
        img=Testing_pixel[i].reshape(48,48,1).astype(np.float32)
      #  img = img[:, :, np.newaxis]
        # img=np.concatenate((img,img,img),axis=-1)
        
        # detail augmentation
        b = cv2.GaussianBlur(img, (3, 3), 0)
        b = b[:, :, np.newaxis]
        img_argument = (img - b) + img        
        # img_argument = np.clip(D, 0, 255).astype(np.uint8)

        # canny extract
        img_gray = img.astype(np.uint8)
        img_canny = cv2.Canny(img_gray, 1, 200)
        img_canny = img_canny[:, :, np.newaxis] + img
        # img_canny = np.clip(img_canny, 0, 255).astype(np.uint8)
        
        img = np.concatenate((img, img_canny, img_argument), axis=-1).astype(np.float32)
        img=np.clip(img, 0, 255).astype(np.uint8)
        # img=np.clip(D1, 0, 255)
        img_all=get_clip_img(img)
        img_all=torch.from_numpy(np.transpose(img_all,(0,3,1,2))/255.)
        img_all=img_all.type(torch.FloatTensor)
    #     # img = tv_F.to_tensor(s/255.)
        print('img=',img.shape)
        if torch.cuda.is_available():
            # img = torch.unsqueeze(img, 0).cuda()
            img_all = img_all.cuda()
        out = model(img_all)
        pred = F.softmax(out, dim=1)
        # print('pred=',pred)
        pred=torch.mean(pred,dim=0,keepdim=True)
        # print('pred=', pred)
        predicted = torch.argmax(pred,dim=1)

        print('label=',label)

        pred_np=predicted.cpu().numpy()[0]
        print('predicted.cpu().numpy()[0]',pred_np)
        if pred_np==label:
            equal_number+=1
        print('face_emotion=',labels_dict[str(pred_np)])
    print('acc=',equal_number/len(Testing_label))

def inference_omg():
    import h5py
    import numpy as np
    model = load_checkpoint('./models_omg/checkpoint.pth')
    print(model)
    path='./data/OMG/OMG_val_data.h5'
    data = h5py.File(path, 'r')
    print(data.keys())
    for i in data.keys():
        print(i)
    Testing_label = data['Testing_label']
    print(Testing_label.shape)
    print(len(Testing_label))
    Testing_pixel = data['Testing_pixel']
    print(Testing_pixel.shape)

    equal_number=0
    for i,label in enumerate(Testing_label):
        img=Testing_pixel[i].reshape(128,128,3).astype(np.float32)
        img=torch.from_numpy(np.transpose(img,(2,0,1))/255.)
        print(img.shape)

        if torch.cuda.is_available():
            model=model.cuda()
            img = torch.unsqueeze(img, 0).cuda()
        out = model(img)
        pred = F.softmax(out, dim=1)
        # print('pred=',pred)
        predicted = torch.argmax(pred, 1)
        # print('predicted=',predicted)
        print('label=',label)

        pred_np=predicted.cpu().numpy()[0]
        print('predicted.cpu().numpy()[0]',pred_np)
        if pred_np==label:
            equal_number+=1
        # print('face_emotion=',labels_dict[str(pred_np)])
    print('acc=',equal_number/len(Testing_label))

def inference_single_img():
    model = load_checkpoint('./models_omg/checkpoint.pth')
    if torch.cuda.is_available():
        model = model.cuda()
    labels_dict={'0':'Anger','1':'Disgust','2':'Fear','3':'Happy','4':'Neutral','5':'Sad','6':'Surprise'}
    
    path='./data/test/other'
    # dirs_list_path=[os.path.join(path,i) for i in os.listdir(path)]
    # # print(dirs_list_path)
    # for dir_list_path in dirs_list_path:
    imgs_list_path=[os.path.join(path,i) for i in os.listdir(path)]
    print(imgs_list_path)
    for img_list_path in imgs_list_path:
        print('Processing image: ' + img_list_path)
        img=cv2.imread(img_list_path)
        img=cv2.resize(img,(128,128))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=np.transpose(img,(2,0,1)).astype(np.float32)/255.
        print(img.shape)
        img=torch.from_numpy(img)

        # img=Image.open(img_list_path)
        # img = tv_F.to_tensor(tv_F.resize(img, (128, 128)))

        if torch.cuda.is_available():
            img = torch.unsqueeze(img, 0).cuda()
        out = model(img)
        pred = F.softmax(out, dim=1)
        print('pred=',pred)
        predicted = torch.argmax(pred, 1)
        # print('predicted=',predicted)
        print('face_emotion=',labels_dict[str(predicted.cpu().numpy()[0])])

def inference_CK():
    path='./data/CK/CK_data.h5'
    model = load_checkpoint('./models_fer2013_aug/model.pth')
    data = h5py.File(path, 'r')
    print(data.keys())
    for i in data.keys():
        print(i)
    Testing_pixel = data['data_pixel']
    print(Testing_pixel.shape)
    Testing_label = data['data_label']
    print(Testing_label.shape)

    if torch.cuda.is_available():
        print('cuda available')
        model = model.cuda()

    equal_number=0
    for i, label in enumerate(Testing_label):
        img = Testing_pixel[i].reshape(48, 48, 1).astype(np.float32)
        s = np.concatenate((img, img, img), axis=-1)
        s=cv2.resize(s,(96,96))
        b1 = cv2.GaussianBlur(s, (3, 3), 0)
        D1 = ((s - b1) + s)
        img_all = get_clip_img(D1)
        img_all = torch.from_numpy(np.transpose(img_all, (0, 3, 1, 2)) / 255.)
        if torch.cuda.is_available():
            # img = torch.unsqueeze(img, 0).cuda()
            img_all = img_all.cuda()
        out = model(img_all)
        pred = F.softmax(out, dim=1)
        # print('pred=',pred)
        pred = torch.mean(pred, dim=0, keepdim=True)
        # print('pred=', pred)
        predicted = torch.argmax(pred, dim=1)

        print('label=', label)

        pred_np = predicted.cpu().numpy()[0]
        print('predicted.cpu().numpy()[0]', pred_np)
        if pred_np == label:
            equal_number += 1

    print('acc=', equal_number / len(Testing_label))

def inference_jaffe():
    model = load_checkpoint('./models_fer2013_aug_deepsort/model.pth')
    if torch.cuda.is_available():
        model = model.cuda()
    labels_dict = {'Anger':0,'Disgust':1,'Fear':2,'Happy':3,'Sad':4,'Surprise':5, 'Neutral':6}

    path = './data/jaffe'
    dirs_list_path=[os.path.join(path,i) for i in os.listdir(path)]
    equal_number=0
    len_img=0
    for dir_list_path in dirs_list_path:
        imgs_list_path = [os.path.join(dir_list_path, i) for i in os.listdir(dir_list_path)]
        len_img+=len(imgs_list_path)
        label=labels_dict[dir_list_path.split('/')[-1]]
        # print(imgs_list_path)
        for img_list_path in imgs_list_path:

            print('Processing image: ' + img_list_path)
            img = cv2.imread(img_list_path)
            s = cv2.resize(img, (48, 48)).astype(np.float32)
            b1 = cv2.GaussianBlur(s, (3, 3), 0)
            D1 = ((s - b1) + s)
            img_all = get_clip_img(D1)
            img_all = torch.from_numpy(np.transpose(img_all, (0, 3, 1, 2)) / 255.)
            if torch.cuda.is_available():
                # img = torch.unsqueeze(img, 0).cuda()
                img_all = img_all.cuda()
            out = model(img_all)
            pred = F.softmax(out, dim=1)
            # print('pred=',pred)
            pred = torch.mean(pred, dim=0, keepdim=True)
            # print('pred=', pred)
            predicted = torch.argmax(pred, dim=1)

            print('label=', label)

            pred_np = predicted.cpu().numpy()[0]
            print('predicted.cpu().numpy()[0]', pred_np)
            if pred_np == label:
                equal_number += 1
    print(equal_number)
    print(len_img)
    print('acc=', equal_number / len_img)
if __name__ == '__main__':
    #test val h5
    # inference_omg_val_h5()
    # inference_single_img()
    inference_fer2013_3channels()
    # inference_omg()
    # inference_fer2013_batchsize()
    # inference_CK()
    # inference_jaffe()

# from torchvision.models import resnet50
# from keras.applications import ResNet50



