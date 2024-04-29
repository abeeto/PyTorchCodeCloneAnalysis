import torch
import torch.nn.functional as F
import numpy as np

def Loss(rpn_score, gt_rpn_score, rpn_loc, gt_rpn_loc, gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc):
    # RPN classifier의 Loss
    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

    # RPN에서 object score가 0보다 큰 anchor 찾음
    pos = gt_rpn_score > 0
    # rpn_loc와 차원을 맞춰서 매칭 시킴
    mask = pos.unsqueeze(1).expand_as(rpn_loc)

    # object score가 0보다 큰 anchor(RPN regressros layer 거친)들의 위치 정보
    mask_loc_preds = rpn_loc[mask].view(-1, 4)
    # object score가 0보다 큰 anchor(고정 크기)들의 위치 정보
    mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

    # 두 위치 정보의 차이를 구함(RPN의 regressors layer 학습)
    x = torch.abs(mask_loc_targets - mask_loc_preds)
    # 주어진 공식에 따라 loss값 구함(변형된 smooth L1 Loss)
    rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))

    # cls와 reg의 가중치를 비슷하게 해주는 하이퍼 파라미터
    rpn_lambda = 10.
    # mask 개수(평균 구함)
    N_reg = (gt_rpn_score > 0).float().sum()
    rpn_loc_loss = rpn_loc_loss.sum() / N_reg
    # 최종 RPN에서의 Loss
    rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)

    gt_roi_loc = torch.from_numpy(gt_roi_locs)
    gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
    # Detector에서의 roi에 대한 loss
    roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)

    n_sample = roi_cls_loc.shape[0]
    # fc layer를 거치며 합쳐진 roi_cls_loc의 위치정보를 다시 roi당, object당 4개로 분배함
    roi_loc = roi_cls_loc.view(n_sample, -1, 4)
    # 각 roi마다의 예측된 label의 위치만 뽑음
    roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]

    # RPN과 같은 방식으로 위치에 대한 Loss 구함
    x = torch.abs(roi_loc - gt_roi_loc)
    roi_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))

    roi_lambda = 10.
    N_reg = (gt_rpn_score > 0).float().sum()
    roi_loc_loss = roi_loc_loss.sum() / n_sample
    # 최종 ROI(Detector) Loss
    roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)

    # 최종 Loss = RPN + ROI
    total_loss = rpn_loss + roi_loss


    return total_loss