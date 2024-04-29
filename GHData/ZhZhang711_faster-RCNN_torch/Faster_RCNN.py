import cupy as cp
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import preprocess
from models.VGG16 import VGG16
from models.RPN import RPN
from models.RCNN import RCNN
from utils.config import config
from utils import converter
from models.utils.boundingbox import t_encoded2bbox
from models.utils.nms.non_maximum_suppression import non_maximum_suppression as nms


def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
            return f(*args,**kwargs)
    return new_f


class Faster_RCNN(nn.Module):
    def __init__(self,
                 n_class=20,
                 extractor_pretrained=True,
                 anchor_ratio=[0.5, 1, 2],
                 anchor_scale=[8, 16, 32],
                 rcnn_init_mean=0, rcnn_init_std=0.01,
                 reg_normalize_mean=(0.0, 0.0, 0.0, 0.0),
                 reg_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()

        self.n_class = n_class + 1
        if config.extractor is 'VGG16':
            self.extractor = VGG16(pretrained=extractor_pretrained)
        else:
            raise NotImplementedError('currently only support VGG16')
        self.RPN = RPN('VGG16', anchor_ratio, anchor_scale)
        self.RCNN = RCNN(n_class + 1, rcnn_init_mean, rcnn_init_std)

        self.reg_normalize_mean = reg_normalize_mean
        self.reg_normalize_std = reg_normalize_std

        self.nms_thresh = 0.3
        self.cls_thresh = 0.05

    def forward(self, img, img_size, img_scale, phase):
        feat = self.extractor(img)
        rpn_cls, rpn_reg, roi_list, roi_id, anchors = self.RPN(feat, img_size, img_scale)
        cls, reg = self.RCNN(feat, roi_list, roi_id)

        # (roi_per_img, n_class \and\ n_class * 4 \and\ 4 \and\ (NA))
        return cls, reg, roi_list, roi_id

    def use_preset(self, preset):
        if preset is 'eval':
            self.nms_thresh = 0.3
            self.cls_thresh = 0.05
        elif preset == 'visualize':
            self.nms_thresh = 0.3
            self.cls_thresh = 0.7
        else:
            raise NotImplementedError('currently only eval and visualize preset is available.')

    def _suppress(self, pred_bbox_np, prob_np):
        # inputs (roi_per_img, n_class * 4), (roi_per_img, n_class)
        bbox = list()
        label = list()
        cls_prob = list()
        for class_ in range(1, self.n_class):  # 0 is bg
            pred_bbox_ = pred_bbox_np.reshape((-1, self.n_class, 4))[:, class_, :]
            prob_ = prob_np[:, class_]

            mask = prob_ > self.cls_thresh
            pred_bbox_ = pred_bbox_[mask]
            prob_ = prob_[mask]

            keep = nms(cp.array(pred_bbox_), self.nms_thresh, prob_)
            keep = cp.asnumpy(keep)

            bbox.append(pred_bbox_[keep])
            label.append(class_ * np.ones(len(keep)) - 1)  # class from 0 to n_class - 2, no bg
            cls_prob.append(prob_[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        cls_prob = np.concatenate(cls_prob, axis=0).astype(np.float32)
        return bbox, label, cls_prob

    @nograd
    def predict(self, img, size=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prep_img = list()
            size = list()
            for img_ in img:
                size_ = img_.shape[1:]
                img_ = preprocess(converter.to_numpy(img_))
                prep_img.append(img_)
                size.append(size_)
        else:
            prep_img = img

        bbox = list()
        label = list()
        cls_prob = list()
        for img_, size_ in zip(prep_img, size):
            img_ = converter.to_tensor(img_[None]).float()
            scale = img_.shape[3] / size_[1]  # (1, C, H, W)[3] = W (no batch here)

            # cls: (roi_per_img, n_class), reg: (roi_per_img, n_class * 4), rois_np: (roi_per_img, 4)
            # tensor, tensor, ndarray
            roi_cls, roi_reg, rois_np, _ = self.forward(img_, size_, scale, 'eval')
            roi_reg = roi_reg.data
            roi = converter.to_tensor(rois_np) / scale

            # both (n_class, 4), t.Tensor on cuda
            mean = t.Tensor(self.reg_normalize_mean).cuda().repeat(self.n_class)[None]
            std = t.Tensor(self.reg_normalize_std).cuda().repeat(self.n_class)[None]

            roi_reg = roi_reg * std + mean
            # reg -> view(roi_per_img, n_class, 4)
            roi_reg = roi_reg.view(-1, self.n_class, 4)
            # roi -> view(roi_per_img, 1, 4) -> expand_as(roi_per_img, n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_reg)
            pred_bbox = t_encoded2bbox(roi.contiguous().view((-1, 4)), roi_reg.view((-1, 4)))
            pred_bbox = pred_bbox.contiguous().view(-1, self.n_class * 4)  # (roi_per_img, n_class * 4)

            pred_bbox[:, 1::2] = (pred_bbox[:, 1::2]).clamp(min=0, max=size_[0])  # clip height
            pred_bbox[:, 0::2] = (pred_bbox[:, 0::2]).clamp(min=0, max=size_[1])  # clip width

            prob_np = converter.to_numpy(F.softmax(roi_cls, dim=1))
            pred_bbox_np = converter.to_numpy(pred_bbox)

            bbox_, label_, cls_prob_ = self._suppress(pred_bbox_np, prob_np)
            bbox.append(bbox_)
            label.append(label_)
            cls_prob.append(cls_prob_)

        self.use_preset('eval')
        self.train()
        return bbox, label, cls_prob

    def get_optimizer(self):
        lr = config.lr
        params = []
        for name, param in dict(self.named_parameters()).items():
            if param.requires_grad:
                if 'bias' in name:
                    params += [{'params': [param], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [param], 'lr': lr, 'weight_decay': config.weight_decay}]

        if config.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
