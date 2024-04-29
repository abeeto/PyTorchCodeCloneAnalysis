import pwd
from turtle import width
from cv2 import CAP_PROP_XI_ACQ_BUFFER_SIZE, sqrt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2


def default_box_generator(layers, large_scale, small_scale):
    '''
    purpose: 
    generate default SSD boxes
    
    input:
    layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    output:
    boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    explanation:
    create an numpy array "boxes" to store default bounding boxes
    you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    the first dimension means number of cells, 10*10+5*5+3*3+1*1
    the second dimension 4 means each cell has 4 default bounding boxes.
    their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    '''
    
    def clip(coordinate):
        '''
        Clamp points out of the image size
        '''
        if coordinate < 0:
            return 0
        elif coordinate > 1:
            return 1
        else:
            return coordinate

    cell_num = np.array(layers)**2
    boxes = np.zeros([np.sum(cell_num),len(large_scale),8],dtype=np.float32)
    index_start = np.append([0],cell_num)
    sqrt2 = np.sqrt(2)
    for box in range(len(layers)):
        length = 1.0/layers[box]
        start = np.sum(index_start[:box+1])
        for x in range(1,layers[box]+1):
            for y in range(1,layers[box]+1):
                for i in range(4):
                    x_center = (x - 0.5) * length
                    y_center = (y - 0.5) * length
                    if i == 0:
                        box_width = box_height = small_scale[box]
                    elif i == 1:
                        box_width = box_height = large_scale[box]
                    elif i == 2:
                        box_width = large_scale[box] * sqrt2
                        box_height = large_scale[box] / sqrt2
                    else:
                        box_width = large_scale[box] / sqrt2
                        box_height = large_scale[box] * sqrt2
                    x_min = clip(x_center - box_width/2.0)
                    x_max = clip(x_center + box_width/2.0)
                    y_min = clip(y_center - box_height/2.0)
                    y_max = clip(y_center + box_height/2.0)
                    boxes[(x-1)*layers[box]+y-1+start,i,:] = [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]    
    return boxes.reshape(np.sum(cell_num)*len(large_scale),8)


def iou(boxs_default, x_min,y_min,x_max,y_max):
    '''
    input:
    boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    x_min,y_min,x_max,y_max -- another box (box_r)
    
    output:
    ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    '''
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    '''
    input:
    ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    boxs_default            -- [num_of_boxes,8], default bounding boxes
    threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    cat_id                  -- class id, 0-cat, 1-dog, 2-person
    x_min,y_min,x_max,y_max -- bounding box

    output:
    ann_box                 -- corresponding SSD boxes w.r.t ground truth bbox in relative coordinates
    ann_confidence          -- class labels of SSD boxes
    img_name                -- image name to produce txt file name
    height_origin           -- the original height of the image (before resize)
    width_origin            -- the original width of the image (before resize)
    '''
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    gx = (x_max + x_min) / 2.0
    gy = (y_max + y_min) / 2.0
    gw = x_max - x_min
    gh = y_max - y_min

    confidence = [0,0,0,0]
    confidence[cat_id] = 1
    indices = np.where(ious_true == True)[0]
    for i in indices:
        # default bbox: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
        px = boxs_default[i,0]
        py = boxs_default[i,1]
        pw = boxs_default[i,2]
        ph = boxs_default[i,3]
        # if px<0 or py<0 or pw<0 or ph<0:
        #     print('p',px,py,pw,ph)
        ann_box[i,:] = [(gx-px)/pw, (gy-py)/ph, np.log(gw/pw), np.log(gh/ph)]
        # print((gx-px)/pw, (gy-py)/ph, np.log(gw/pw), np.log(gh/ph))
        ann_confidence[i,:] = confidence
    
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if len(indices) == 0:
        ious_index = np.argmax(ious)
        px = boxs_default[ious_index,0]
        py = boxs_default[ious_index,1]
        pw = boxs_default[ious_index,2]
        ph = boxs_default[ious_index,3]

        ann_box[ious_index,:] = [(gx-px)/pw, (gy-py)/ph, np.log(gw/pw), np.log(gh/ph)]
        ann_confidence[ious_index,:] = confidence

    return  ann_box,ann_confidence



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320, augmentation=False):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5 #0.6 0.7 0.8
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        self.augmentation = augmentation
        
        #split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if self.anndir and self.train == True:
            self.img_names = self.img_names[:round(len(self.img_names)*0.9)]
        elif self.anndir and self.train == False:
            self.img_names = self.img_names[round(len(self.img_names)*0.9):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"

        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = cv2.imread(img_name)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        if self.anndir: #self.train:
            class_id,x_min,y_min,x_max,y_max = [], [], [], [], []
            with open(ann_name,'r') as f:
                for line in f:
                    c_id,x_s,y_s,w_box,h_box = line.split()
                    c_id = int(c_id)
                    x_s = float(x_s)
                    y_s = float(y_s)
                    w_box = float(w_box)
                    h_box = float(h_box)

                    class_id.append(c_id) 
                    x_min.append(x_s)  
                    y_min.append(y_s) 
                    x_max.append(x_s + w_box) 
                    y_max.append(y_s + h_box) 
            class_id,x_min,y_min,x_max,y_max = np.asarray(class_id), np.asarray(x_min), np.asarray(y_min), np.asarray(x_max), np.asarray(y_max)

        if self.augmentation: #self.train and 
            random_x_min = np.random.randint(0, np.min(x_min),size=1)[0]
            random_y_min = np.random.randint(0, np.min(y_min),size=1)[0]
            random_x_max = np.random.randint(np.max(x_max)+1,width,size=1)[0]
            random_y_max = np.random.randint(np.max(y_max)+1,height,size=1)[0]

            x_min -= random_x_min
            y_min -= random_y_min
            x_max -= random_x_min
            y_max -= random_y_min
            
            width = random_x_max - random_x_min
            height = random_y_max - random_y_min
            image = image[random_y_min:random_y_max,random_x_min:random_x_max,:]
            

        if self.anndir: #self.train:
            for i in range(len(class_id)):
                ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,\
                                                class_id[i],x_min[i]/width,y_min[i]/height,x_max[i]/width,y_max[i]/height)
        
        image_preprocess = transforms.Compose([transforms.ToTensor(),transforms.Resize([self.image_size,self.image_size])])
        image  = image_preprocess(image)
        ann_box = torch.from_numpy(ann_box)
        ann_confidence = torch.from_numpy(ann_confidence)
        
        return image, ann_box, ann_confidence, img_name[-9:-4], height, width

if __name__ == '__main__':
    boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
    dataset = COCO("data/train/images/", "data/train/annotations/", 4, boxs_default, train = True, image_size=320)
    # dataset_test = COCO("data/train/images/", "data/train/annotations/", 4, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=0)

    dataset_test = COCO("data/test/images/", "", 4, boxs_default, train = False, image_size=320)
    print(dataset_test.__len__())
    # images_, ann_box_, ann_confidence_, img_name, height, width = dataset_test.__getitem__(33)
    # # for i, data in enumerate(dataloader, 0):
    # #     images_, ann_box_, ann_confidence_,img_name,height,width = data
    # #     print(height)
    #     # images_ = images_.numpy()
    # # images_ = images_ * 255
    # image = torch.permute(images_, (1,2,0))
    # print(image.shape)
    # print(img_name)
    # #     # cv2.imwrite("%d"%i+".jpg",image)
    # # import matplotlib.pyplot as plt
    # # plt.imshow(cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2RGB))
    # # plt.show()
    # from utils import *
    # visualize_pred('test', ann_confidence_.numpy(), ann_box_.numpy(),  ann_confidence_.numpy(), ann_box_.numpy(), images_.numpy(), boxs_default)
    