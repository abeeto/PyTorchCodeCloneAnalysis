from NMS import NMS
import numpy as np
import torch
import torch.nn as nn

def Detector(ground_truth_boxes, labels, roi, img_feature, pred_anchor_locs, pred_cls_scores, anchor_locations, anchor_labels):

    n_sample = 64 # roi에서 sample 할 개수
    pos_ratio = 0.25 # sample에서 pos한 개수
    pos_iou_thresh = 0.5 # pos로 간주되기 위해 ground truth box와 겹치는 비율
    neg_iou_thresh_hi = 0.5 # neg로 간주되기 위해 back ground와 겹치는 비율(high&low)
    neg_iou_thresh_lo = 0.0

    # roi와 ground truth box의 iou
    ious = np.empty((len(roi), len(ground_truth_boxes)), dtype=np.float32) # (150, len(ground_truth_boxes))
    ious.fill(0)

    for num1, i in enumerate(roi):
        xa1, ya1, xa2, ya2 = i
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, j in enumerate(ground_truth_boxes):
            xb1, yb1, xb2, yb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)
            # 겹치는 box 좌표를 구한다.
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            # 제대로 된 box(겹치는 부분이 존재하는)일 경우 iou 구한다.
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = iter_area / (anchor_area + box_area - iter_area)            
            else:
                iou = 0.
            # roi와 ground truth box의 iou 저장
            ious[num1, num2] = iou
            
    # gt_assignment: 각 anchor당 iou가 최대인 index
    gt_assignment = ious.argmax(axis=1) # (150,)
    # max_iou: 각 anchor당 iou가 최대인 값
    max_iou = ious.max(axis=1) # (150,)

    # gt_assignment의 index와 실제 label을 매칭
    gt_roi_label = labels[gt_assignment] # (150,)
    # max_iou가 pos_iou_thresh(0.5) 이상인 것만 선택
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    # 이미지 당 pos roi의 개수
    pos_roi_per_image = np.round(n_sample * pos_ratio) # 16
    # pos_index를 고려해 이미지 당 pos roi 개수 설정
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    # pos_index 중 pos_roi_per_this_image만큼 중복허용하지 않고 random하게 뽑는다.
    if pos_roi_per_this_image > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

    # neg에 대해서도 같은 작업을 한다.
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_roi_per_this_image > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

    # pos, neg의 index를 저장
    keep_index = np.append(pos_index, neg_index) # (64,)
    # anchor의 label을 저장
    gt_roi_labels = gt_roi_label[keep_index] # (64,)
    # negative 부분의 label은 0으로 설정
    gt_roi_labels[pos_roi_per_this_image:] = 0
    # sample_roi: 최종적으로 선택된 n_sample개의 anchor
    sample_roi = roi[keep_index] # (64,4)

    # 선택된 anchor의 y(정답)값
    gt_boxes_for_sampled_roi = ground_truth_boxes[gt_assignment[keep_index]] # (64, 4)

    # anchor의 위치정보와 gt box의 위치정보
    height = sample_roi[:, 3] - sample_roi[:, 1]
    width = sample_roi[:, 2] - sample_roi[:, 0]
    ctr_x = sample_roi[:, 0] + 0.5 * height
    ctr_y = sample_roi[:, 1] + 0.5 * width
    base_height = gt_boxes_for_sampled_roi[:, 3] - gt_boxes_for_sampled_roi[:, 1]
    base_width = gt_boxes_for_sampled_roi[:, 2] - gt_boxes_for_sampled_roi[:, 0]
    base_ctr_x = gt_boxes_for_sampled_roi[:, 0] + 0.5 * base_height
    base_ctr_y = gt_boxes_for_sampled_roi[:, 1] + 0.5 * base_width

    # feature map에서 위치를 찾기 위해 공식대로 위치정보를 가공한다.(RPN과 동일)
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose() # (64, 4)

    # sample_roi를 CNN에 넣기 위한 작업
    rois = torch.from_numpy(sample_roi).float() # (64, 4)
    # roi의 index 또한 tensor형으로 만든다.
    roi_indices = 0 * np.ones((len(rois),), dtype=np.int32) # (64)
    roi_indices = torch.from_numpy(roi_indices).float()

    # roi와 해당 index를 concatenate를 통해 이어붙인다.
    # [:, None]: 1xN인 행렬을 Nx1로 바꾸는 효과
    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1) # (64, 5) (indice, x1, y1, x2, y2)
    indices_and_rois = indices_and_rois.contiguous()

    # Pooling size
    # 논문상 (7, 7)이지만(image: 800x800) 해당 프로젝트의 image size(224, 224)를 고려하여 낮춤
    size = (2, 2)
    # Pooling layer
    adaptive_max_pool = nn.AdaptiveMaxPool2d((size[0], size[1]))
    output = []
    # rois를 가져와 feature map에 맞게 좌표들을 16으로 나눔
    rois = indices_and_rois.data.float()
    rois[:, 1:].mul_(1/16.0)
    # rois를 slicing하기 위해 정수형으로 변환
    rois = rois.long()
    num_rois = rois.size(0) # sample 개수

    for i in range(num_rois):
        # 한 sample   [, x1:(x2)+1, y1:(y2)+1]
        roi = rois[i]
        # narrow(a, b, c): tensor의 a차원의 [b:c] 부분을 포인터로 가져온다.(변경 시 원본도 변경 됨)
        # feature map에서 anchor부분에 해당하는 부분만 MaxPooling한다.
        # [..., ]: [:,]에서 :는 처음 차원의 전체 선택인 반면, ...는 뒤에 나오는 index들을 행렬의 뒤에서부터 slicing한 후
        # 나머지 앞의 전체 선택이다.
        im = img_feature[..., roi[1]:(roi[3]+1), roi[2]:(roi[4]+1)] # [, x1:(x2+1), y1:(y2+1)]
        output.append(adaptive_max_pool(im))
    # MaxPooling된 feature들을 concatenate한다.
    output = torch.cat(output, 0) # (64, 512, 2, 2)
    # fc layer에 넣기 위해 행렬을 일렬로 핀다.
    fc_output = output.view(output.size(0), -1) # (64, 2048)

    # RoI Pooling을 거친 feature을 두 fc layer에 병렬적으로 넣는다.
    roi_head_classifier = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 512)])
    cls_loc = nn.Linear(512, 21 * 4) # (object:20 + 배경:1) * (좌표 정보:4)

    cls_loc.weight.data.normal_(0, 0.01)
    cls_loc.bias.data.zero_()

    score = nn.Linear(512, 21) # (object:20 + 배경:1)

    fc_output = roi_head_classifier(fc_output)
    # 64개 roi에 대한 좌표값
    roi_cls_loc = cls_loc(fc_output) # (64, 84)
    # 64개 roi에 대한 object score값
    roi_cls_score = score(fc_output) # (64, 21)

    # 입출력을 위해 tensor을 조금 변형한다.(빈 차원 제거)
    rpn_loc = pred_anchor_locs[0] # (1764, 4)
    rpn_score = pred_cls_scores[0] # (1764, 2)
    # gt_rpn: RPN에서의 anchor(tighting 전)
    gt_rpn_loc = torch.from_numpy(anchor_locations) # (1764, 4)
    gt_rpn_score = torch.from_numpy(anchor_labels) # (1764)


    return rpn_score, gt_rpn_score, rpn_loc, gt_rpn_loc, gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc