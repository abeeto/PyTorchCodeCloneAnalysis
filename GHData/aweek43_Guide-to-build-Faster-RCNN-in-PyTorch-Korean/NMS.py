from RPN import RPN
import numpy as np

def NMS(anchors, pred_anchor_locs, objectness_score):

    # Non-Maximum Suppression(NMS): ground truth box와 iou가 0.7이상 겹치는 anchor들이 많이 나오기 때문에
    # 하나만 남기고 나머지는 지운다.
    nms_thresh = 0.7 # NMS를 위한 임계값
    n_train_pre_nms = 900 # train 시 NMS에 넣기 전 뽑는 objectness가 높은 순서의 anchor 개수
    n_train_post_nms = 150 # train 시 NMS로 뽑은 anchor 수
    n_test_pre_nms = 450 # test 시 NMS에 넣기 전 뽑는 objectness가 높은 순서의 anchor 개수
    n_test_post_nms = 25 # test 시 NMS로 뽑은 anchor 수
    min_size = 16 # anchor의 최소 높이

    # anchor의 왼쪽 위, 오른쪽 아래 좌표로부터 중심점과 높이, 너비를 구한다
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

    # reg layer를 거친 anchor의 location을 구하기 위해 numpy로 변환
    pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy() # (1764, 2)
    # objectness_score를 계산을 위해 numpy로 변환
    objectness_score_numpy = objectness_score[0].data.numpy() # (1764)

    # pred_anchor의 정보를 구분한다. 각 원소마다 배열로 만들어 저장한다.(dx: 2차원 배열)
    dx = pred_anchor_locs_numpy[:, 0::4]
    dy = pred_anchor_locs_numpy[:, 1::4]
    dh = pred_anchor_locs_numpy[:, 2::4]
    dw = pred_anchor_locs_numpy[:, 3::4]
    # 상대적 좌표를 다시 원본 이미지에 매치되는 좌표로 복구한다.
    # newaxis: 원소 하나하나를 배열화 시킨다.(배열 1차원 증가)
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    # regions of interest 배열 선언
    roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchors.dtype) # (1764, 4)
    # roi에 pred_anchor의 왼쪽위, 오른쪽아래 좌표를 저장한다.
    roi[:, 0::4] = ctr_x - 0.5 * w # format: {x1,y1,x2,y2}
    roi[:, 1::4] = ctr_y - 0.5 * h
    roi[:, 2::4] = ctr_x + 0.5 * w
    roi[:, 3::4] = ctr_y + 0.5 * h

    img_size = (224, 224) #Image size
    # clip(a, b, c): 배열 a의 원소 중 b미만은 b, c 이상은 c로 바꾼다.
    # slice(a, b, c): for _ in range(a, b, c)
    # 좌표가 이미지 사이즈를 넘어갈 경우 최대(or 최소)로 바꾼다.
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

    # 최소 사이즈보다 작은 anchor들을 버린다.
    ws = roi[:, 2] - roi[:, 0]
    hs = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :] # (1764, 4)
    # 범위 안에 든 anchor들의 objectness_score만 뽑는다.
    score = objectness_score_numpy[keep] # (1764,)

    # ravel(): 다차원 배열을 1차원으로 펼침
    # argsort(): 오름차순 정렬된 원소의 index를 반환
    # [::-1]: 역순으로 반환
    # order: score가 높은 순서대로 1차원배열로 index를 저장한다.
    order = score.ravel().argsort()[::-1] # (1764,)
    # score가 높은 순서로 n_train_pre_nms개만큼 뽑음
    order = order[:n_train_pre_nms] # (900,)
    # 해당 anchor를 저장
    # sorted_roi = roi[order, :] # (900, 4)

    # NMS를 위해 anchor끼리 겹치는 비율을 구해 제거한다.
    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]
    # anchor의 넓이
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        # 겹치는 anchor 중 첫번재 anchor만 저장
        keep.append(i)
        # x1 기준으로 x1좌표가 더 큰 좌표만 계산한다.(a,b의 계산이 b,a에서 다시 계산되는 것 방지)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # x1,x2(y1, y2)가 서로 역전되지 않는 경우를 생각한다.
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # ovr: anchor끼리의 iou
        ovr = inter / (area[i] + area[order[1:]] - inter)
        # ovr가 임계값(0.7)이 안 넘는 index
        inds = np.where(ovr <= nms_thresh)[0]
        # order에서 inds와 기준 anchor(i)를 뺀다.
        order = np.setdiff1d(order, inds)
        order = np.setdiff1d(order, i)

    # 최종 roi를 구한다.
    keep = keep[:n_train_post_nms] # while training/testing , use accordingly
    roi = roi[keep] # the final region proposals

    return roi, pred_anchor_locs