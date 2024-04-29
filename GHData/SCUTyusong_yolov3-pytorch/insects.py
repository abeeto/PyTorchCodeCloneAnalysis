import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from anchor import *

INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus',
                'acuminatus', 'armandi', 'coleoptera', 'linnaeus']


def get_insect_names():
    """
    return a dict, as following,
        {'Boerner': 0,
         'Leconte': 1,
         'Linnaeus': 2,
         'acuminatus': 3,
         'armandi': 4,
         'coleoptera': 5,
         'linnaeus': 6
        }
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


# cname2cid = get_insect_names()
# print(cname2cid)


# canemcid 是有关昆虫和类别对应的字典，datadir是文件夹路径，返回records，带有所有图片的标注信息
def get_annotations(cname2cid, datadir):
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))

    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]  # fid is the number
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)  # fpath was the path of fname
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')  # 是一个路径
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs),), dtype=np.int32)
        is_crowd = np.zeros((len(objs),), dtype=np.int32)
        difficult = np.zeros((len(objs),), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            ################ 这里使用xywh格式来表示目标物体真实框 #############
            gt_bbox[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1 + 1., y2 - y1 + 1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


# TRAINDIR = './insects/train/'
# TESTDIR = './insects/test'
# VALIDDIR = './insects/val'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record=records[0]
# print(records[0])


def get_bbox(gt_bbox, gt_class):  # 将图片真实边界框bbox和真实标签labels输入，将行数拓展到50输出
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2


def get_img_data_from_file(record):  # 输出图片矩阵，真实边界框矩阵，标签矩阵，宽度，高度
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    im_file = record['im_file']  # 获取图片文件路径
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), \
        "image height of {} inconsistent in record({}) and img file({})".format(
            im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
        "image width of {} inconsistent in record({}) and img file({})".format(
            im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)

# TRAINDIR = './insects/train/'
# TESTDIR = './insects/test'
# VALIDDIR = './insects/val'
# cname2cid = get_insect_names()
# records = get_annotations(cname2cid, TRAINDIR)
# record = records[0]
# img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
# print(img.shape)
# print(gt_labels)
# print(scales)
# print(img.shape)
# plt.imshow(img)
# currentAxis = plt.gca()
# gtbbox=record['gt_bbox']
# # gtbbox1=gtbbox[0]
# draw_rectangle_xywh(currentAxis, gtbbox[4])
# print(gt_labels)
# print(scales)
# plt.show()
