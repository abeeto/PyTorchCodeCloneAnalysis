import numpy as np
import torch

def Soft_non_max_suppression(prediction, num_classes, conf_thres=0.5, sigma=0.5, nms_thres=0.4):
    # bs = np.shape(prediction)[0]
    # 将框转换成左上角右下角的形式
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    # output = []
    output = [None for _ in range(len(prediction))]
    # 1、对所有图片进行循环。
    for image_i, image_pred in enumerate(prediction):
        # prediction = prediction[i]
        # 2、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
        mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[mask]
        if not image_pred.size(0):
            continue

        # 3、判断第2步中获得的框的种类与得分。
        # 取出预测结果中框的位置与之进行堆叠。
        # 此时最后一维度里面的内容由5+num_classes变成了4+1+2，
        # 四个参数代表框的位置，一个参数代表预测框是否包含物体，两个参数分别代表种类的置信度与种类。
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # class_conf = np.expand_dims(np.max(prediction[:, 5:5 + num_classes], 1), -1)
        # class_pred = np.expand_dims(np.argmax(prediction[:, 5:5 + num_classes], 1), -1)
        # detections = np.concatenate((prediction[:, :5], class_conf, class_pred), 1)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # unique_class = np.unique(detections[:, -1])
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        # if len(unique_labels) == 0:
        #     continue

        # 4、对种类进行循环，
        # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
        # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
        for c in unique_labels:
            cls_mask = detections[:, -1] == c
            best_box = []
            detection = detections[cls_mask]
            _, conf_sort_index = torch.sort(detection[:, 4], descending=True)
            # scores = detection[:, 4]
            # 5、根据得分对该种类进行从大到小排序。
            # arg_sort = np.argsort(scores)[::-1]
            detection = detection[conf_sort_index]
            # print(detection)
            while detection.size(0):
                best_box.append(detection[0].unsqueeze(0))
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1], detection[1:])
                # 将获得的IOU取高斯指数后乘上原得分，之后重新排序
                detection[1:, 4] = torch.exp(-(ious * ious) / sigma) * detection[1:, 4]
                detection = detection[1:]
                scores = detection[:, 4]
                arg_sort = torch.argsort(scores,dim=-1)
                detection = detection[arg_sort]
            # output.append(best_box)
            max_detections = torch.cat(best_box).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))
    return output ,prediction


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou




# import numpy as np
#
#
# def non_max_suppression(boxes, num_classes, conf_thres=0.5, sigma=0.5, nms_thres=0.4):
#     bs = np.shape(boxes)[0]
#     # 将框转换成左上角右下角的形式
#     shape_boxes = np.zeros_like(boxes[:, :, :4])
#     shape_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
#     shape_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
#     shape_boxes[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
#     shape_boxes[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
#
#     boxes[:, :, :4] = shape_boxes
#     output = []
#     # 1、对所有图片进行循环。
#     for i in range(bs):
#         prediction = boxes[i]
#         # 2、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
#         mask = prediction[:, 4] >= conf_thres
#         prediction = prediction[mask]
#         if not np.shape(prediction)[0]:
#             continue
#
#         # 3、判断第2步中获得的框的种类与得分。
#         # 取出预测结果中框的位置与之进行堆叠。
#         # 此时最后一维度里面的内容由5+num_classes变成了4+1+2，
#         # 四个参数代表框的位置，一个参数代表预测框是否包含物体，两个参数分别代表种类的置信度与种类。
#         class_conf = np.expand_dims(np.max(prediction[:, 5:5 + num_classes], 1), -1)
#         class_pred = np.expand_dims(np.argmax(prediction[:, 5:5 + num_classes], 1), -1)
#         detections = np.concatenate((prediction[:, :5], class_conf, class_pred), 1)
#         unique_class = np.unique(detections[:, -1])
#
#         if len(unique_class) == 0:
#             continue
#
#         best_box = []
#         # 4、对种类进行循环，
#         # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
#         # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
#         for c in unique_class:
#             cls_mask = detections[:, -1] == c
#
#             detection = detections[cls_mask]
#             scores = detection[:, 4]
#             # 5、根据得分对该种类进行从大到小排序。
#             arg_sort = np.argsort(scores)[::-1]
#             detection = detection[arg_sort]
#             print(detection)
#             while np.shape(detection)[0] > 0:
#                 best_box.append(detection[0])
#                 if len(detection) == 1:
#                     break
#                 ious = iou(best_box[-1], detection[1:])
#                 # 将获得的IOU取高斯指数后乘上原得分，之后重新排序
#                 detection[1:, 4] = np.exp(-(ious * ious) / sigma) * detection[1:, 4]
#                 detection = detection[1:]
#                 scores = detection[:, 4]
#                 arg_sort = np.argsort(scores)[::-1]
#                 detection = detection[arg_sort]
#         output.append(best_box)
#     return np.array(output)
#
#
# def iou(b1, b2):
#     b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
#     b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
#
#     inter_rect_x1 = np.maximum(b1_x1, b2_x1)
#     inter_rect_y1 = np.maximum(b1_y1, b2_y1)
#     inter_rect_x2 = np.minimum(b1_x2, b2_x2)
#     inter_rect_y2 = np.minimum(b1_y2, b2_y2)
#
#     inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
#                  np.maximum(inter_rect_y2 - inter_rect_y1, 0)
#
#     area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#     area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#
#     iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
#     return iou
#
