import torch
import torchvision
import torch.nn as nn
import numpy as np

# RPN: 이미지를 CNN에 넣어 나온 feature map의 픽셀들로 anchor들을 만든다.
# ground truth boxes와 labels을 이용해 각 anchor마다 점수를 매기고
# reg layer, classifier layer에 병렬적으로 넣는다.

def RPN(sample_image, ground_truth_boxes, labels):

    ###########################################################################################
    # vgg16 모델 로드
    model = torchvision.models.vgg16(pretrained=True)
    # 뒤의 fc layer(classifier)은 빼고 feature 추출 부분만 가져옴
    model = list(model.features)

    # CNN통과 후 feature map size
    feature_size = 224//16 # 14
    # features layer에서 마지막 pooling layer는 빼야하기 때문에 pooling 전 사이즈일때까지만 image를 layer에 통과시킴
    img_feature = sample_image.clone()
    for layer in model:
        img_feature = layer(img_feature)
        if img_feature.size()[2] <= feature_size:
            break
    ###########################################################################################

    # anchor 비율 설정(1:1, 1:2, 2:1)
    ratios = [0.5, 1, 2]
    # anchor sclae 설정(16배로 작아진 feature에서 1픽셀=원본image의 16픽셀)
    anchor_scales = [2, 4, 8]
    # 각 feature map의 픽셀마다 9개 anchor이 생기고 각 anchor마다 4개(box 좌표)가 생김
    anchors = np.zeros((feature_size * feature_size * 9, 4), dtype=np.float32) # (1764, 4)

    # 원본image의 중심 크기
    sub_sample = 16
    # anchor들의 중심 좌표(원본 image는 feature map에서 16배하므로)
    ctr_x = np.arange(8, 16 * feature_size - 7, 16)
    ctr_y = np.arange(8, 16 * feature_size - 7, 16)

    # 2차원인 x, y 중심점들을 1차원으로 이어붙인다.(for문을 하나 줄이는 효과)
    ctr = np.array([[0]*2 for _ in range(feature_size * feature_size)]) # (196, 2)
    index = 0
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 0] = ctr_x[x]
            ctr[index, 1] = ctr_y[y]
            index +=1

    # anchor들의 좌표를 구해 저장한다
    index = 0
    for c in ctr:
        ctr_y, ctr_x = c
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                # 각 anchor scale에 따른 box의 h, w
                # 넓이 비가 0.5:1:2가 되기 위해 ratios[i]의 루트를 곱함
                h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
                # 각 anchor의 왼쪽 위 좌표와 오른쪽 아래 좌표를 저장한다(format: x1,y1,x2,y2)
                anchors[index, 0] = ctr_x - w / 2.
                anchors[index, 1] = ctr_y - h / 2.
                anchors[index, 2] = ctr_x + w / 2.
                anchors[index, 3] = ctr_y + h / 2.
                index += 1

    # image 내에 있는 anchor의 index를 뽑는다.
    inside_index = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 224) &
            (anchors[:, 3] <= 224)
    )[0] # (792)

    # inside_index와 같은 크기의 배열을 -1로 채운다. anchor의 점수를 매긴다.
    anchor_label = np.empty((len(inside_index), ), dtype=np.int32)
    anchor_label.fill(-1)

    # image 내에 있는 anchor만 따로 뽑는다.
    valid_anchors = anchors[inside_index] # (792, 4)

    # iou: anchor box와 ground-truth box가 겹치는 부분의 비율
    ious = np.empty((len(valid_anchors), len(ground_truth_boxes)), dtype=np.float32) # (792, 2)
    ious.fill(0)

    for anchor_index, i in enumerate(valid_anchors):
        xa1, ya1, xa2, ya2 = i  
        anchor_area = (ya2 - ya1) * (xa2 - xa1) # anchor의 넓이
        for ground_truth_box_index, j in enumerate(ground_truth_boxes):
            xb1, yb1, xb2, yb2 = j
            box_area = (yb2- yb1) * (xb2 - xb1) # ground-truth-box의 넓이
            # anchor와 ground-truth-box가 겹치는 사각형을 찾는다. 
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                # 겹치는 부분의 넓이
                overlap_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = overlap_area / (anchor_area + box_area - overlap_area)            
            else:
                iou = 0.
            ious[anchor_index, ground_truth_box_index] = iou

    # 각 x축(열)에서 최대값의 index를 찾는다.(gt_box마다 iou의 최대값의 index)
    # 찾은 index의 value 값을 찾는다
    gt_max_ious = ious.max(axis=0) # [gt1_max, gt2_max, ...]

    # 각 y축(행)에서 최대값의 index를 찾는다.(anchor마다 iou의 최대값의 index)
    argmax_ious = ious.argmax(axis=1) # (792,)
    # 찾은 index의 value 값을 찾는다
    max_ious = ious[np.arange(len(inside_index)), argmax_ious]

    # 2차원 numpy array의 where(): return값의 첫번째 array는 행들, 두번째 array는 열들이다
    # gt_max_ious와 같은 iou을 가지는 곳들의 행들을(anchor index) 찾는다
    gt_argmax_ious = np.where(ious == gt_max_ious)[0] # [anchor_index1, anchor_index2, ...]

    # iou가 0.7이상이면 1, 0.3미만이면 0을 부여한다.(그 사이값은 무시)
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    # 해당 anchor의 모든 gt_box에 대한 iou중 최대가 0.3미만이면 해당 anchor_label을 0으로 바꾼다.
    anchor_label[max_ious < neg_iou_threshold] = 0
    # gt_box 중 가장 큰 iou를 가진 anchor에 1을 부여한다.
    anchor_label[gt_argmax_ious] = 1
    # 해당 anchor의 모든 gt_box에 대한 iou중 최대가 0.7이상이면 해당 anchor_label을 1으로 바꾼다.
    anchor_label[max_ious >= pos_iou_threshold] = 1

    # pos한 anchor_label비율, batch size
    pos_ratio = 0.5
    n_sample = 256
    # pos한 anchor 개수
    n_pos = pos_ratio * n_sample
    pos_anchor_index = np.where(anchor_label == 1)[0]
    # pos_anchor이 n_pos보다 클 경우 넘는 개수는 random하게 -1로 바꾼다.
    if len(pos_anchor_index) > n_pos:
        disable_index = np.random.choice(pos_anchor_index, size=(len(pos_anchor_index) - n_pos), replace=False)
        anchor_label[disable_index] = -1

    # pos_anchor을 제외한 만큼 neg_anchor을 뽑는다.
    n_neg = n_sample - np.sum(anchor_label == 1)
    neg_anchor_index = np.where(anchor_label == 0)[0]
    if len(neg_anchor_index) > n_neg:
        disable_index = np.random.choice(neg_anchor_index, size=(len(neg_anchor_index) - n_neg), replace = False)
        anchor_label[disable_index] = -1

    # anchor마다 가장 iou가 높은 gt_box의 좌표를 저장한다.
    max_iou_bbox = ground_truth_boxes[argmax_ious] # (792, 4)
    # valid_anchor, max_iou_bbox의 좌표(높이, 너비, 중심)(feature map에서의 좌표)
    # 기존 저장되있던 왼쪽 위, 오른쪽 아래 좌표로도 직사각형을 표현할 수 있으나
    # 논문상에 나온대로 (높이,너비,중심점)을 이용해 표현하기 위한 과정
    height = valid_anchors[:, 3] - valid_anchors[:, 1]
    width = valid_anchors[:, 2] - valid_anchors[:, 0]
    ctr_x = valid_anchors[:, 0] + 0.5 * height
    ctr_y = valid_anchors[:, 1] + 0.5 * width
    base_height = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
    base_width = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
    base_ctr_x = max_iou_bbox[:, 0] + 0.5 * base_height
    base_ctr_y = max_iou_bbox[:, 1] + 0.5 * base_width
    
    # eps: 파라미터의 자료형 중 가장 작은 값
    eps = np.finfo(height.dtype).eps
    # 계산상의 값을 명확하게 해줌(Machine epsilon)
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    # anchor의 상대적인 좌표를 구한다.(network에 입력하고 loss을 구할수 있도록)
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    # vstack()은 열의 수가 같은 배열을 밑으로 이어붙인다.
    # transpose()는 (i,j)원소와 (j,i)를 바꾸는 전치를 수행한다.
    # 열로 구분되면 하나의 anchor를 행으로 구분하게 함으로써 코드상으로 이해가 쉽다.
    anchor_locs = np.vstack((dx, dy, dh, dw)).transpose() # (792, 4)
    # 전체 anchor에 대한 label을 정의한다.
    anchor_labels = np.empty((len(anchors),), dtype=anchor_label.dtype)
    anchor_labels.fill(-1)
    anchor_labels[inside_index] = anchor_label # (1764,)
    # 전체 anchor에 대한 위치 좌표를 정의한다.
    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[inside_index, :] = anchor_locs # (1764, 4)

    in_channels = 512 # RPN의 feature map의 output channel(VGG16일 경우 512)
    mid_channels = 512
    n_anchor = 9 # 각 pixel당 anchor의 수

    conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
    # fc layer에서 fc가 아닌 kernel_size=1인 conv를 이용한다.
    # 각 anchor당 4개의 좌표
    reg_layer = nn.Conv2d(mid_channels, n_anchor*4, kernel_size=1, stride=1, padding=0) # regressors
    # 각 anchor당 2개의 값(object or not)
    cls_layer = nn.Conv2d(mid_channels, n_anchor*2, kernel_size=1, stride=1, padding=0) # classifier(object or not)

    # layer에 대해 가중치는 평균 0, 분산 0.01, 바이어스는 0으로 초기화한다.
    conv1.weight.data.normal_(0, 0.01)
    conv1.bias.data.zero_()
    reg_layer.weight.data.normal_(0, 0.01)
    reg_layer.bias.data.zero_()
    cls_layer.weight.data.normal_(0, 0.01)
    cls_layer.bias.data.zero_()

    # RPN에서 나온 feature map을 2개의 fc layer에 넣는다
    x = conv1(img_feature) # (1, 512, 14, 14)
    pred_anchor_locs = reg_layer(x) # (1, 36, 14, 14)
    pred_cls_scores = cls_layer(x) # (1, 18, 14, 14)

    # permute(a,b,c): tensor의 첫번째 차원은 a번째 차원으로, 두번째 차원은 b번째, 세번째 차원은 c번째 차원으로 바꾼다.
    # contiguous(): tensor을 실제 메모리상에 연속되게 위치시킨다. 축을 변화시키며 메모리상에 불연속적으로 위치되기 때문이다.
    # view(): 파라미터의 모양대로 tensor를 재조합한다.(-1은 가변적으로 남은 tensor값들을 넣는다.)
    # pred_anchor_locs: 모든 anchor(1764개)의 좌표값(4개)
    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4) # (1, 1764, 4)
    # pred_cls_scores: 모든 pixel(14x14개)의 anchor(9개)의 classifier값(2개)
    pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous() # (1, 14, 14, 18)
    # view(1, feature_size, feature_size, 9, 2): 각 anchor당 object인지 아닌지 두 종류로 나눈다.
    # [:, :, :, :, 1]: 마지막 차원의 2번째 값이 object일 확률이다.
    # view(1, -1): 한 열로 합쳐 모든 anchor들의 object일 확률을 나타낸다.
    # objectness_score: 모든 anchor들의 object일 확률
    objectness_score = pred_cls_scores.view(1, feature_size, feature_size, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1) # (1, 1764)
    # pred_cls_scores: 모든 anchor들의 object일 확률과 아닐 확률
    pred_cls_scores  = pred_cls_scores.view(1, -1, 2) # (1, 1764, 2)

    return ground_truth_boxes, labels, img_feature, anchors, pred_anchor_locs, pred_cls_scores, anchor_locations, anchor_labels, objectness_score