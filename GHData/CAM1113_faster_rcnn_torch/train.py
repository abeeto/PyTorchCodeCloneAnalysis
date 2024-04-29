import torch
from torch.utils.data.dataloader import DataLoader
from ImageDataset import ImageDataSet
import numpy as np
import random
from torch import optim
from net.LossHelper import cls_loss, smooth_l1, cls_loss_detector, regress_loss_detector
from net.Models import Res50, RPN, Detector, RoIPooling
from utils.Anchors import get_anchors
from utils.BBoxUtility import BBoxUtility
from utils.Config import Config
from torch.autograd import Variable

from utils.RoIHelpers import calc_iou

NUM_CLASSES = 20
dataset = ImageDataSet()
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
config = Config()
bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap, ignore_threshold=config.rpn_min_overlap)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
back_bone = Res50().to(device)
rpn = RPN(len(config.anchor_box_scales) * len(config.anchor_box_ratios)).to(device)
roIPooling = RoIPooling().to(device)
detector = Detector(num_class=NUM_CLASSES).to(device)

lr = 1e-5
optimizer = optim.Adam(rpn.parameters(), lr=lr)
optimizer2 = optim.Adam(back_bone.parameters(), lr=lr)
optimizer3 = optim.Adam(detector.parameters(), lr=lr)


def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width), get_output_length(height)


def get_examples_rpn(image, y):
    y = y.float().numpy()
    batch, _, height, width = image.shape
    if len(y) == 0:
        print("no object in ground truth")
        return None

    boxes = np.array(y[:, :4], dtype=np.float)
    boxes[:, 0] = boxes[:, 0] / width
    boxes[:, 1] = boxes[:, 1] / height
    boxes[:, 2] = boxes[:, 2] / width
    boxes[:, 3] = boxes[:, 3] / height

    box_heights = boxes[:, 3] - boxes[:, 1]
    box_widths = boxes[:, 2] - boxes[:, 0]
    if (box_heights <= 0).any() or (box_widths <= 0).any():
        print("box size error")
        return None

    y[:, :4] = boxes[:, :4]
    anchors = get_anchors(get_img_output_length(width, height), width, height)

    # 获取训练的框,assignment是二维的numpy向量，锚框个数 * 5
    # 5代表RPN输出的边框偏移量，和锚点框 忽略框（-1），背景框（0），包含目标框（1）
    assignment = bbox_util.assign_boxes(y, anchors)

    # 一张图片由256个锚框参与训练
    num_regions = 256
    classification = assignment[:, 4]
    regression = assignment[:, :]

    # 正样本下标
    mask_pos = classification[:] > 0
    # 正样本总数
    num_pos = len(classification[mask_pos])

    if num_pos > num_regions / 2:
        # 正样本数大于256的一半（125），则把多余的正样本忽略掉
        val_locs = random.sample(range(num_pos), int(num_pos - num_regions / 2))

        mask_pos = np.array(mask_pos.nonzero()).reshape(-1)

        for i in val_locs:
            index = mask_pos[i]
            classification[index] = -1
            regression[index][-1] = -1
        # 赋值失败
        # classification[mask_pos][val_locs] = -1
        # regression[mask_pos][val_locs, -1] = -1
        # print(classification[mask_pos][val_locs])
        num_pos = num_regions / 2

    # 负样本下标（负样本是背景）
    mask_neg = classification[:] == 0
    # 负样本总数
    num_neg = len(classification[mask_neg])
    if num_neg + num_pos > num_regions:
        # 负样本总数+正样本总数大于256，随机忽略掉一部分负样本
        val_locs = random.sample(range(num_neg), int(num_neg - num_regions + num_pos))
        mask_neg = np.array(mask_neg.nonzero()).reshape(-1)
        # classification[mask_neg][val_locs] = -1
        for i in val_locs:
            index = mask_neg[i]
            classification[index] = -1

    classification = np.reshape(classification, [-1, 1])
    regression = np.reshape(regression, [-1, 5])
    return [np.expand_dims(np.array(classification, dtype=np.float), 0),
            np.expand_dims(np.array(regression, dtype=np.float), 0), y]


# P_rpn rpn网络预测的结果，height 图片的高，width图片的宽，boxes所有的标注框（0-3是位置，4是类别）
def get_example_classify(P_rpn, height, width, boxes):
    anchors = get_anchors(get_img_output_length(width, height), width, height)
    # 将预测结果进行解码
    results = bbox_util.detection_out(P_rpn, anchors, confidence_threshold=0)
    # 所有框的左上和右下坐标集合，小数形式

    R = results[0]
    print("R = {}".format(R.shape))
    X2, Y1, Y2 = calc_iou(R, config, boxes, width, height, NUM_CLASSES)
    if X2 is None or Y1 is None or Y2 is None:
        print("X2 is None or Y1 is None or Y2 is None")
        return None
    print("X2 = {}".format(X2))
    print("Y1 = {}".format(Y1))
    print("Y2 = {}".format(Y2))

    # neg_samples、pos_samples，正负样本下标
    neg_samples = np.where(Y1[0, :, 0] == NUM_CLASSES)
    pos_samples = np.where(Y1[0, :, 0] != NUM_CLASSES)
    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []
    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []
    if len(neg_samples) == 0:
        return None
    print("neg_samples.shape= {}".format(neg_samples))
    print("pos_samples.shape= {}".format(pos_samples))

    # 正负样本均衡
    if len(pos_samples) < config.num_rois // 2:
        selected_pos_samples = pos_samples.tolist()
    else:
        selected_pos_samples = np.random.choice(pos_samples, config.num_rois // 2, replace=False).tolist()
    try:
        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                replace=False).tolist()
    except:
        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                replace=True).tolist()

    sel_samples = selected_pos_samples + selected_neg_samples
    X2 = X2[:, sel_samples, :]
    Y1 = Y1[:, sel_samples, :]
    Y2 = Y2[:, sel_samples, :]
    return X2, Y1, Y2


total_epoch = 1000


def train():
    init_loss = 100
    for epoch in range(total_epoch):
        for index, data in enumerate(dataLoader):
            # image 是图片  y是左上、右下、类别
            image, y = data
            image = Variable(image).to(device)
            y = y[0]
            example = get_examples_rpn(image, y)
            if example is None:
                continue
            feature = back_bone(image)
            class_rpn, regress_rpn = rpn(feature)
            print("regress_rpn = {}".format(regress_rpn))
            class_examples = Variable(torch.from_numpy(example[0])).to(device)
            regress_examples = Variable(torch.from_numpy(example[1])).to(device)
            loss_class = cls_loss(class_examples, class_rpn)
            loss_regress = smooth_l1(regress_examples, regress_rpn)
            loss = loss_class + loss_regress
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            print("loss rpn = {}".format(loss.item()))

            feature = back_bone(image).clone().detach()
            P_rpn = rpn(feature)
            P_rpn = (P_rpn[0].cpu().detach().numpy(), P_rpn[1].cpu().detach().numpy())
            print(P_rpn[1])
            batch, channels, height, width = image.shape
            # example[2],标注框的小数形式
            example_classify = get_example_classify(P_rpn, height=height, width=width, boxes=example[2])
            if example_classify is None:
                print("example_classify is None")
                continue
            X2, Y1, Y2 = example_classify

            X2 = Variable(torch.from_numpy(X2).float().to(device))
            Y1 = Variable(torch.from_numpy(Y1.reshape(-1)).long().to(device))
            Y2 = Variable(torch.from_numpy(Y2.reshape(-1, Y2.shape[2])).float().to(device))
            rois = roIPooling((feature, X2))
            classify, regress = detector(rois)

            detector_classify_loss = cls_loss_detector(y_pred=classify, y_true=Y1)
            print("regress  = {}".format(regress))
            detector_regress_loss = regress_loss_detector(y_true=Y2, y_pred=regress, label=Y1)
            print("detector_regress_loss = {}".format(detector_regress_loss.item()))
            loss_detector = detector_classify_loss + detector_regress_loss
            optimizer3.zero_grad()
            loss_detector.backward()
            optimizer3.step()

            print("detector_classify_loss.item() = {}".format(detector_classify_loss.item()))

            if loss.item() < init_loss:
                torch.save(back_bone, "back_bone.pth")
                torch.save(rpn, "rpn.pth")
                torch.save(roIPooling, "roIPooling.pth")
                torch.save(detector, "detector.pth")
                init_loss = loss.item()
        print("epoch:{}".format(epoch))


if __name__ == '__main__':
    train()
