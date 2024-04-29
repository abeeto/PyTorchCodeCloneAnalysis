import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math


### in this py, four function was defined, it were 'draw_retangle','draw_anchor_box','box_iou_xyxy'
# 定义画矩形框的程序    此函数的作用即为输入box，currentAxis,画出一个矩形
def draw_rectangle_xyxy(currentAxis, bbox, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标（数值最小的坐标）、矩形区域的宽度、高度等参数
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, linewidth=1,
                             edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


def draw_rectangle_xywh(currentAxis, bbox, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, w, h] center: x1,y1
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标（数值最小的坐标）、矩形区域的宽度、高度等参数.
    rect = patches.Rectangle((bbox[0] - bbox[2] / 2 + 1, bbox[1] - bbox[3] / 2 + 1), bbox[2], bbox[3], linewidth=1,
                             edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


# plt.figure(figsize=(10, 15))
# # pdb.set_trace()
#
# filename = './insects/train/images/1.jpeg'
# im = imread(filename)
# plt.imshow(im)
# # 使用xyxy格式表示物体真实框
# bbox1 = [214.29, 325.03, 399.82, 631.37]
# bbox2 = [40.93, 141.1, 226.99, 515.73]
# bbox3 = [247.2, 131.62, 480.0, 639.32]
# bbox4 = [300, 300, 200, 400]
# bbox5=[]
# currentAxis = plt.gca()
#
# draw_rectangle_xyxy(currentAxis, bbox1, edgecolor='r')
# draw_rectangle_xyxy(currentAxis, bbox2, edgecolor='r')
# draw_rectangle_xyxy(currentAxis, bbox3, edgecolor='r')
# draw_rectangle_xywh(currentAxis, bbox4)
# plt.show()


# 绘制锚框 锚框数量为len(scales)*len(ratios),且画出的锚框中心相同
def draw_anchor_box(currentAxis, center, length, scales, ratios, img_height, img_width):
    """
    以center为中心，产生一系列锚框
    其中length指定了一个基准的长度
    scales是包含多种尺寸比例的list
    ratios是包含多种长宽比的list
    img_height和img_width是图片的尺寸，生成的锚框范围不能超出图片尺寸之外
    """
    bboxes = []
    for scale in scales:
        for ratio in ratios:
            h = length * scale * math.sqrt(ratio)
            w = length * scale / math.sqrt(ratio)
            x1 = max(center[0] - w / 2., 0.)
            y1 = max(center[1] - h / 2., 0.)
            x2 = min(center[0] + w / 2. - 1.0, img_width - 1.0)  # 若锚框边缘超出了图片，则以图片为界
            y2 = min(center[1] + h / 2. - 1.0, img_height - 1.0)
            print(center[0], center[1], w, h)
            bboxes.append([x1, y1, x2, y2])

    for bbox in bboxes:
        draw_rectangle_xyxy(currentAxis, bbox, edgecolor='b')


# filename = './insects/train/images/1.jpeg'
# im = imread(filename)
# img_height = im.shape[0]
# img_width = im.shape[1]
#
# plt.imshow(im)
# currentAxis = plt.gca()
# draw_anchor_box(currentAxis, [300., 500.], 100., [2.0, 4.0], [0.5, 1.0, 2.0], img_height, img_width)
# plt.show()


# ################# 以下为添加文字说明和箭头###############################
#
# plt.text(285, 285, 'G1', color='red', fontsize=20)
# plt.arrow(300, 288, 30, 40, color='red', width=0.001, length_includes_head=True, \
#           head_width=5, head_length=10, shape='full')
#
# plt.text(190, 320, 'A1', color='blue', fontsize=20)
# plt.arrow(200, 320, 30, 40, color='blue', width=0.001, length_includes_head=True, \
#           head_width=5, head_length=10, shape='full')
#
# plt.text(160, 370, 'A2', color='blue', fontsize=20)
# plt.arrow(170, 370, 30, 40, color='blue', width=0.001, length_includes_head=True, \
#           head_width=5, head_length=10, shape='full')
#
# plt.text(115, 420, 'A3', color='blue', fontsize=20)
# plt.arrow(127, 420, 30, 40, color='blue', width=0.001, length_includes_head=True, \
#           head_width=5, head_length=10, shape='full')
#
# # draw_anchor_box([200., 200.], 100., [2.0], [0.5, 1.0, 2.0])
# plt.show()


# 计算IoU，矩形框的坐标形式为xyxy，这个函数会被保存在box_utils.py文件中
def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w

    # 计算相并面积
    union = s1 + s2 - intersection
    # pdb.set_trace()
    # 计算交并比
    iou = intersection / union
    return iou


# 计算IoU，矩形框的坐标形式为xywh
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    x1max, y1max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    x2max, y2max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou

# bbox1 = [100., 100., 100., 100.]
# bbox2 = [120., 120., 100., 100.]
# iou = box_iou_xywh(bbox1, bbox2)
# print('IoU is {}'.format(iou))
