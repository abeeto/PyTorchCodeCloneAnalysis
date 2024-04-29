import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt
from matplotlib.image import imread
from insects import *


# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
    # fromarray : transform array to image
    # as array : transfrom image to array
    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img


# filename = './insects/train/images/1.jpeg'
# img = imread(filename)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# img = random_distort(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()


# 随机填充
def random_expand(img,
                  gtboxes,
                  max_ratio=4.,
                  fill=None,
                  keep_ratio=True,
                  thresh=0.5):
    if random.random() > thresh:
        return img, gtboxes

    if max_ratio < 1.0:
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)  # 生成一个随机实数，范围在1和max_ratio之间
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)  # 经过这步操作之后，ratio_x与ratio_y的值保持一致
    oh = int(h * ratio_y)  # oh,ow是将图片扩充后的图片高和宽
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)  # 产生一个介于0，ow-w之间的随机整数
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c))  # 创立一个oh,ow,c的三维矩阵
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    out_img[off_y:off_y + h, off_x:off_x + w, :] = img  # 将原始图片至于新生成图片的随机一点off_x,off_y处
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes


# filename = './insects/train/images/771.jpeg'
# img = imread(filename)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
#
# cname2cid = get_insect_names()
# TRAINDIR = './insects/train/'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record = records[0]
# img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
# img,gtboxes = random_expand(img,gt_boxes)
# # img=np.asarray(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()


# 计算交并比，如果box1与box2之间shape一致，则可计算。（box1与box2均为矩阵）。假如box1与box2中均有5个边界框，则返回5个iou值
def multi_box_iou_xywh(box1, box2):
    """
    In this case, box1 or box2 can contain multi boxes.
    Only two cases can be processed in this method:
       1, box1 and box2 have the same shape, box1.shape == box2.shape
       2, either box1 or box2 contains only one box, len(box1) == 1 or len(box2) == 1
    If the shape of box1 and box2 does not match, and both of them contain multi boxes, it will be wrong.
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2  # b1_x1,b1_x2为box1的坐上角坐标
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)  # 保留inter_w中大于0的数值
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


# record=records[2]

# bbox1=records[2]['gt_bbox']
# bbox2=records[1]['gt_bbox']
# a=multi_box_iou_xywh(bbox1,bbox2)
# print(a)
def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (
            boxes[:, 0] + boxes[:, 2] / 2) * im_w  # boxes第一列变成x1，第二列变成x2
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (
            boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])  # 定义一个[x,y,x+w,y+h]位置范围内的矩形
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0  # ?
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(
        axis=1)  # 逻辑与

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (
            boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (
            boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


# 随机裁剪
def random_crop(img,
                boxes,
                labels,
                scales=[0.3, 1.0],
                max_ratio=2.0,
                constraints=None,
                max_trial=50):
    if len(boxes) == 0:
        return img, boxes

    if not constraints:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0),
                       (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w,
                                  (crop_y + crop_h / 2.0) / h,
                                  crop_w / float(w), crop_h / float(h)]])

            iou = multi_box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels


# cname2cid = get_insect_names()
# TRAINDIR = './insects/train/'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record = records[0]
# img = record['im_file']
# img1 = imread(img)
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# bbox = record['gt_bbox']
# gt_labels = record['gt_class']
# img2, boxes, labels = random_crop(img1, bbox, gt_labels)
# plt.subplot(1, 2, 2)
# plt.imshow(img2)
# plt.show()

# 随机缩放 size(float) is the size after random_interp
def random_interp(img, size, interp=None):
    interp_method = [
        cv2.INTER_NEAREST,  # 最近邻插值，
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)  # fx fy are the scale ratio
    return img


# 随机翻转 ::-1 was to left-right flip or up-down flip
def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[::-1, :, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes


# 随机打乱真实框排列顺序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]


# cname2cid = get_insect_names()
# TRAINDIR = './insects/train/'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record = records[10]
# gtboxes = record['gt_bbox']
# gtlabel=record['gt_class']
# gt_boxes,gt_labels=shuffle_gtbox(gtboxes,gtlabel)
# print(gt_boxes,gt_labels)
# img = record['im_file']
# size = record['w']
# img1 = imread(img)
# img2, gt_boxes1 = random_flip(img1, gtboxes)
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# plt.subplot(1, 2, 2)
# plt.imshow(img2)
# print(img1.shape)
# print(img2.shape)
# plt.show()

# 图像增广方法汇总
def image_augment(img, gtboxes, gtlabels, size, means=None):
    # 随机改变亮暗、对比度和颜色等
    img = random_distort(img)
    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    # 随机裁剪
    img, gtboxes, gtlabels, = random_crop(img, gtboxes, gtlabels)
    # 随机缩放
    img = random_interp(img, size)
    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)
    # 随机打乱真实框排列顺序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')


def get_img_data(record, size=640):
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img, gt_boxes, gt_labels, scales


# cname2cid = get_insect_names()
# TRAINDIR = './insects/train/'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record = records[10]
# img1 = record['im_file']
# img1 = imread(img1)
# # img2, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
# size = 512
# img2, gt_boxes, gt_labels,scales = get_img_data(record, size=480)
#
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# plt.subplot(1, 2, 2)
# plt.imshow(img2)
# print(img1.shape)
# print(img2.shape)
# plt.show()
