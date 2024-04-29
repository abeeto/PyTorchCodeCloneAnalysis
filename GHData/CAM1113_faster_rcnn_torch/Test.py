import numpy as np
import torch
import torchvision.transforms as transforms
from net.Models import Res50, RPN, RoIPooling
from train import get_img_output_length
from utils.Anchors import get_anchors
from utils.ImageUtil import loadImage, draw_rect_mul
from utils.BBoxUtility import BBoxUtility
from utils.Config import Config
from torchvision import ops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
back_bone = torch.load(r'./back_bone.pth').eval()
rpn = torch.load(r'./rpn.pth').eval()
roIPooling = RoIPooling().to(device).eval()
detector = torch.load(r"./detector.pth").eval()
__transforms__ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
__transforms_noinit__ = [
    transforms.ToTensor(),
]


def getImageName(index):
    file = open("./train.txt", "r")
    file_content = file.read().strip()
    file.close()
    labels = file_content.split("\n")
    return labels[index].split(" ")[0]


def getRois(P_rpn, width, height):
    anchors = get_anchors(get_img_output_length(width, height), width, height)
    # 将预测结果进行解码
    results = bbox_util.detection_out(P_rpn, anchors, confidence_threshold=0)
    # 所有框的左上和右下坐标集合，小数形式
    R = results[0]

    x1 = R[:, 0] * width / config.rpn_stride
    y1 = R[:, 1] * height / config.rpn_stride
    x2 = R[:, 2] * width / config.rpn_stride
    y2 = R[:, 3] * height / config.rpn_stride

    x1 = np.round(x1[:])
    y1 = np.round(y1[:])
    x2 = np.round(x2[:])
    y2 = np.round(y2[:])

    w = x2 - x1
    h = y2 - y1
    box = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1)), axis=1)
    return box


config = Config()
bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap, ignore_threshold=config.rpn_min_overlap)
if __name__ == '__main__':
    fileName = getImageName(0)

    image = loadImage(fileName)
    transform = transforms.Compose(__transforms__)
    image = transform(image)
    _, height, width = image.shape
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2]).to(device)
    feature = back_bone(image)
    P_rpn = rpn(feature)
    P_rpn = (P_rpn[0].cpu().detach().numpy(), P_rpn[1].cpu().detach().numpy())

    box = getRois(P_rpn, width, height)
    X2 = torch.from_numpy(box).float().to(device)
    rois = roIPooling((feature, X2))
    classify, regress = detector(rois)
    regress = regress.view(regress.shape[0], -1, 4)  # batch + 类 + 左上和宽高编码


    objects_list = classify.argmax(axis=1) != config.NUM_CLASSES
    objects_list = objects_list.cpu().numpy().nonzero()[0]
    R = X2.cpu().numpy()[objects_list]
    classify = classify[objects_list].cpu().detach().numpy()
    label = classify.argmax(axis=1)
    regress = regress[objects_list].cpu().detach().numpy()

    print(classify.shape)
    print("regress.shape = {}".format(regress.shape))
    print("label = {}".format(label))
    print('R.shape = {}'.format(R))
    paramater = []
    confidence = []
    for ii in range(regress.shape[0]):
        paramater.append(regress[ii,label[ii],:])
        confidence.append(classify[ii,label[ii]])
    parameter = np.array(paramater,dtype=np.float)
    confidence = np.array(confidence,dtype=np.float)
    print("parameter.shape = {}".format(parameter.shape))
    # high_conf_idx = (confidence > 0.8).nonzero()[0]
    # print(high_conf_idx)
    # parameter = paramater[high_conf_idx,:]
    # confidence = confidence[high_conf_idx]
    # R = R[high_conf_idx,:]
    # label = label[high_conf_idx]


    parameter[:, 0] /= config.classifier_regr_std[0]
    parameter[:, 1] /= config.classifier_regr_std[1]
    parameter[:, 2] /= config.classifier_regr_std[2]
    parameter[:, 3] /= config.classifier_regr_std[3]
    print("parameter = {}".format(parameter))

    print("parameter.shape = {}".format(parameter.shape))
    (x, y, w, h) = R[:,0],R[:,1],R[:,2],R[:,3]
    (tx, ty, tw, th) = parameter[:, 0],parameter[:, 1],parameter[:, 2], parameter[:, 3]
    # 解码
    # 计算建议框的中心
    cx = x + w / 2.
    cy = y + h / 2.
    # 调整后建议框的中心
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    # 调整后建议框的宽和高
    w1 = np.exp(tw) * w
    h1 = np.exp(th) * h
    # 调整后建议框的左上
    x1 = cx1 - w1 / 2.
    y1 = cy1 - h1 / 2.
    # 调整后建议框的右下
    x2 = cx1 + w1 / 2
    y2 = cy1 + h1 / 2
    # 取整
    x1 = np.round(x1).reshape(-1,1)
    y1 = np.round(y1).reshape(-1,1)
    x2 = np.round(x2).reshape(-1,1)
    y2 = np.round(y2).reshape(-1,1)

    boxes_regress = np.concatenate([x1,y1,x2,y2],axis=1)

    boxes_regress = boxes_regress.clip(min=0)
    print(boxes_regress)

    if(boxes_regress.shape[0] == 0):
        print("未检测到目标")

    # 边框转化成小数，之后用来计算其在原图上的位置
    boxes_regress[:, 0] = boxes_regress[:, 0] * config.rpn_stride / width
    boxes_regress[:, 1] = boxes_regress[:, 1] * config.rpn_stride / height
    boxes_regress[:, 2] = boxes_regress[:, 2] * config.rpn_stride / width
    boxes_regress[:, 3] = boxes_regress[:, 3] * config.rpn_stride / height


    # 极大值抑制
    boxes_regress = torch.from_numpy(boxes_regress)
    confidence = torch.from_numpy(confidence)
    nms_result = ops.nms(boxes_regress,confidence,iou_threshold=0.3).numpy()
    print("nms_result = {}".format(nms_result))
    boxes_regress = boxes_regress.numpy()
    confidence = confidence.numpy()
    boxes_regress = boxes_regress[nms_result][:3]
    confidence = confidence[nms_result][:3]
    label = label[nms_result][:3]



    # 求出真实框
    boxes_regress[:, 0] = boxes_regress[:, 0] * width
    boxes_regress[:, 1] = boxes_regress[:, 1] * height
    boxes_regress[:, 2] = boxes_regress[:, 2] * width
    boxes_regress[:, 3] = boxes_regress[:, 3] * height
    image = loadImage(fileName)
    transform = transforms.Compose(__transforms_noinit__)
    image = transform(image)
    draw_rect_mul(image,boxes_regress)
    print(label)
















    # indices_for_object = np.array((class_rpn[:, 0] > 0.7).nonzero()).reshape(-1)
    # print(indices_for_object.shape)
    # anchors = get_anchors(get_img_output_length(width, height), width, height)
    # anchors = anchors[indices_for_object]
    # regress_rpn = regress_rpn.cpu().detach().view(-1,4).numpy()
    # regress_rpn = regress_rpn[indices_for_object]
    # image = loadImage(fileName)
    # transform = transforms.Compose(__transforms_noinit__)
    # image = transform(image)
    # boxes =  bbox_util.decode_boxes(regress_rpn,anchors)
    # print(boxes)
    # print(width,height)
    # boxes[:,0] = boxes[:,0] * width
    # boxes[:,1] = boxes[:,1] * height
    # boxes[:,2] = boxes[:,2] * width
    # boxes[:,3] = boxes[:,3] * height

    # draw_rect_mul(image, boxes)
