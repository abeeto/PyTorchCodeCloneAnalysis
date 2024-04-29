import torch      #pytorch
import torch.nn as nn     #pytorch network
from torch.utils.data import Dataset, DataLoader      #pytorch dataset
from tensorboardX import SummaryWriter     #tensorboard
import torchvision      #torchvision
import torch.optim as optim     #pytorch optimizer
import numpy as np      #numpy

import os     #os

import cv2      #opencv (box 그리기를 할 때 필요)
from PIL import Image     #PILLOW (이미지 읽기)
import time     #time
import imgaug as ia     #imgaug
from imgaug import augmenters as iaa
from torchvision import transforms      #torchvision transform

import json as js
from tqdm import tqdm
from engine import evaluate
import pickle

#GPU연결
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
#device=torch.device('cpu')


print(device)

ROOT_DIR = '../'
DATA_DIR = '../data/'

JSON_DIR = 'D:/dataset/obj_detect_ai_hub/json_test/'
IMG_DIR = 'D:/dataset/obj_detect_ai_hub/img_all/'

class_name_to_id_mapping = {"경찰차": 0,
                            "구급차": 1,
                            "기타특장차(견인차, 쓰레기차, 크레인 등)": 2,
                            "성인(노인포함)": 3,
                            "어린이": 4,
                            "자전거": 5,
                            "오토바이": 6,
                            "전동휠/전동킥보드/전동휠체어": 7,
                            "버스(소형,대형)": 8,
                            "세단": 9,
                            "통학버스(소형,대형)": 10,
                            "트럭": 11,
                            "SUV/승합차": 12}

label_eng_dic = {"경찰차":"Police_Car",
                 "구급차": "Ambulance",
                 "기타특장차(견인차, 쓰레기차, 크레인 등)": "Special_Car",
                 "성인(노인포함)": "Adult/Elder",
                 "어린이": "Child",
                 "자전거": "Bike",
                 "오토바이": "Auto-Bike",
                 "전동휠/전동킥보드/전동휠체어": "Electric_Wheel",
                 "버스(소형,대형)": "Bus",
                 "세단": "Sedan",
                 "통학버스(소형,대형)": "School_Bus",
                 "트럭": "Truck",
                 "SUV/승합차": "SUV"}

label_dic = class_name_to_id_mapping

def json_parser(jfile):
    k = 0
    object_name = []
    bbox = []

    # JSON 파일읽기
    with open(JSON_DIR+jfile, encoding='UTF8') as json_file:
        json_data = js.load(json_file)
    # JSON 내 annotation 배열 설정
    jsonArray = json_data.get("annotations")
    # annotation 배열 열고 읽기
    file_name = jfile.split('.')[0]
    class_id = ''
    for list in jsonArray:
        json_label = list.get("label")
        json_attribute = list.get("attributes")
        json_attr_detail = json_attribute.get(json_label)

        try:
            class_id = class_name_to_id_mapping[json_attr_detail]
        except KeyError:
            #print(json_attribute, json_attr_detail)
            #print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
            pass

        json_bbox = list.get("points")
        one_bbox = []
        np_bbox = np.array(json_bbox)
        xmin = np_bbox[0, 0]
        ymin = np_bbox[0, 1]
        xmax = np_bbox[2, 0]
        ymax = np_bbox[2, 1]

        one_bbox.append(xmin)
        one_bbox.append(ymin)
        one_bbox.append(xmax)
        one_bbox.append(ymax)


        object_name.append(json_attr_detail)
        bbox.append(one_bbox)

    return file_name, object_name, bbox

def makeBox(voc_im, bbox, objects):
    image = voc_im.copy()
    for i in range(len(objects)):
        cv2.rectangle(image,(int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),color = (0,255,0),thickness = 1)
        cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # 크기, 색, 굵기
    return image

def Total_Loss(loss):
    loss_objectness = loss['loss_objectness'].item()
    loss_rpn_box_reg = loss['loss_rpn_box_reg'].item()
    loss_classifier = loss['loss_classifier'].item()
    loss_box_reg = loss['loss_box_reg'].item()

    rpn_total = loss_objectness + 10*loss_rpn_box_reg
    fast_rcnn_total = loss_classifier + 1*loss_box_reg

    total_loss = rpn_total + fast_rcnn_total

    return total_loss


global prev_img
global prev_targets
#
#
#
class AiHub(Dataset):

    def __init__(self, json_list, len_data):
        self.json_list = json_list
        self.len_data = len_data
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.Fliplr(0.5)
        self.resize = iaa.Resize({"shorter-side": 1080, "longer-side": "keep-aspect-ratio"})

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        jfile = str(self.json_list[idx])

        file_name, object_name, bbox = json_parser(jfile)
        num_objs = len(object_name)

        image_path = IMG_DIR+str(file_name)+'.png'
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        #print(file_name, ':', bbox)
        image_id = torch.tensor([idx])
        #
        image, bbox = self.flip(image=image, bounding_boxes=np.array([bbox]))
        image, bbox = self.resize(image=image, bounding_boxes=bbox)
        image = self.to_tensor(image)

        area = (bbox[0][:, 3] - bbox[0][:, 1]) * (bbox[0][:, 2] - bbox[0][:, 0])
        bbox = bbox.squeeze(0).tolist()

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        masks = torch.zeros((num_objs,), dtype=torch.int64)
        targets = []
        d = {}
        d['boxes'] = torch.tensor(bbox, device=device)
        try:
            d['labels'] = torch.tensor([label_dic[x] for x in object_name], dtype=torch.int64, device=device)
        except:
            d['labels'] = torch.tensor([label_dic['세단'] for x in object_name], dtype=torch.int64, device=device)
        d['image_id'] = image_id
        d["masks"] = masks
        d["area"] = area
        d["iscrowd"] = iscrowd
        targets.append(d)

        return image, targets

def run():
    json_list = os.listdir(JSON_DIR)
    # json_list.sort()
    print(json_list)

    label_set = set()
    cnt = 0
    tot_num_json = int(len(json_list))
    for json in tqdm(range(len(json_list))):
        cnt += 1
        file_name, object_name, bbox = json_parser(json_list[json])
        #print(f'{cnt} / {tot_num_json}')

    # #
    # #
    # #
    #데이터 테스트 출력
    # dataloader = DataLoader(dataset, shuffle=True)
    #
    # for i, (image, targets) in enumerate(dataloader):
    #   test_image = image
    #   test_target = targets
    #   if i == 0 : break
    #
    # print(test_target)
    #
    # labels = test_target[0]['labels'].squeeze_(0)
    # objects = []
    # # for lb in labels:
    # #   objects.append([k for k, v in label_dic.items() if v == lb][0])
    #
    # for lb in labels:
    #   objects.append([label_eng_dic[k] for k, v in label_dic.items() if v == lb][0])
    #
    # plot_image = makeBox(test_image.squeeze(0).permute(1,2,0).numpy(),
    #                      test_target[0]['boxes'].squeeze(0), objects
    #                      )
    # plt.imshow(plot_image)
    # plt.show()



    #
    # GPU 메모리 할당 문제
    #
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    '''
    #
    # MODEL 수정 및 다른 backbone 추가해서 돌리기
    #
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
  
    num_classes = len(label_dic)  # 13 개 클래스
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=13)
  
    '''
    backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
    backbone_out = 512
    backbone.out_channels = backbone_out

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),
                                                                        aspect_ratios=((0.5, 1.0, 2.0),))

    resolution = 7
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels=backbone_out * (resolution ** 2),
                                                                   representation_size=4096)
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096, 13)  # 13 class

    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                                                    min_size=600, max_size=1000,
                                                    rpn_anchor_generator=anchor_generator,
                                                    rpn_pre_nms_top_n_train=6000, rpn_pre_nms_top_n_test=6000,
                                                    rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                                                    rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                                    rpn_batch_size_per_image=1, rpn_positive_fraction=0.5,
                                                    box_roi_pool=roi_pooler, box_head=box_head,
                                                    box_predictor=box_predictor,
                                                    box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=300,
                                                    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                                                    box_batch_size_per_image=1, box_positive_fraction=0.25
                                                    )

    loss_sum = 0

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    #
    # training
    #
    total_epoch = 10 # 1번에 1시간 소요

    writer = SummaryWriter("../models/faster_rcnn/runs/faster_rcnn")
    #f_evaluation = open("coco_eval.txt", 'w')

    term = 10
    try:
        check_point = torch.load(ROOT_DIR + "models/faster_rcnn/check_point.pth")
        start_epoch = check_point['epoch']
        start_idx = check_point['iter']
        model.load_state_dict(check_point['state_dict'])
        optimizer.load_state_dict(check_point['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=0.00001,
                                                               last_epoch=start_epoch)
        scheduler.load_state_dict(check_point['scheduler'])


    except:
        print("check point load error!")
        start_epoch = 0
        start_idx = 0



    # print("start_epoch = {} , start_idx = {}".format(start_epoch, start_idx))
    print("Training Start")

    model.train()

    start = time.time()
    #
    # 이부분 training / validation 으로 나누기
    #
    from sklearn.model_selection import train_test_split

    train_json_list, valid_json_list = train_test_split(json_list, test_size=0.3, random_state=42)
    train_dataset = AiHub(train_json_list, len(train_json_list))
    valid_dataset = AiHub(valid_json_list, len(valid_json_list))

    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    validloader = DataLoader(valid_dataset, shuffle=False, batch_size=1)

    print(f'train:valid = {len(dataloader)}:{len(validloader)}')

    for epoch in range(start_epoch, total_epoch):

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        for i, (image, targets) in enumerate(dataloader, start_idx):
            optimizer.zero_grad()

            targets[0]['boxes'].squeeze_(0)
            targets[0]['labels'].squeeze_(0)
            try:
                loss = model(image.to(device), targets)
            except:
                print('loss error!')
                continue

            #print("loss= ",loss)
            total_loss = Total_Loss(loss)
            loss_sum += total_loss
            if (i + 1) % term == 0:
                end = time.time()
                print("Epoch {}/{} | Iter {}/{} | Loss: {} | Duration: {} sec".format(epoch, total_epoch,
                                                                                      (i + 1), len(dataloader),
                                                                                      (loss_sum / term),
                                                                                      round((end - start), 3)))
                writer.add_scalar('Training Loss', loss_sum / term, epoch * len(dataloader) + i)

                state = {
                    'epoch': epoch,
                    'iter': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(state, ROOT_DIR + "models/faster_rcnn/check_point.pth")

                loss_sum = 0
                start = time.time()

        #total_loss.backward()
        optimizer.step()
        start_idx = 0
        scheduler.step()

        state = {
            'epoch': epoch,
            'iter': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, ROOT_DIR + "models/faster_rcnn/check_point.pth")

        #
        # evaluation
        #
        model.eval()
        with torch.no_grad():
            coco_eval = evaluate(model, validloader, device=device)
            with open('path_and_filename.pickle', 'wb') as handle:
                pickle.dump(coco_eval.coco_eval, handle)

        model.train()

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), ROOT_DIR + "models/faster_rcnn/epoch{}.pth".format(epoch))

    writer.close()
    #f_evaluation.close()

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


if __name__ == '__main__':

    run()

