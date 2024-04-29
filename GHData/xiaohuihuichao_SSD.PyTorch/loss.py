import torch
import torch.nn.functional as F

from match import match
from config import config
from prior import get_prior
from iou import jaccard_iou
from box_transform import point_form, center_form, encode


class MultiBoxLoss():
    def __init__(self, config=config):
        self.config = config
        self.priors = None
        self.priors_point = None

    def __call__(self, predictions, targets, neg_rate=3.):
        return self.forward(predictions, targets, neg_rate)

    def forward(self, predictions, targets, neg_rate):
        loc_out, conf_out = predictions
        # loc_target # shape [B, n, 4]
        # conf_target # shape [B, n]
        loc_target, conf_target = targets[:, :, 0:4], targets[:, :, 4].type(torch.int64)
        
        if self.priors_point == None or self.priors == None:
            feature_map_sizes = [l.shape[1:3] for l in loc_out]
            priors = [get_prior(feature_map_size, self.config["image_size"], min_size, max_size, ratio) for feature_map_size, min_size, max_size, ratio in zip(feature_map_sizes, self.config["min_sizes"], self.config["max_sizes"], self.config["ratios"])]
            priors = torch.cat(priors, dim=0) # shape [num_priors_per_img, 4]
            priors_point = point_form(priors)
            if config["cuda"]:
                self.priors = priors.cuda()
                self.priors_point = priors_point.cuda()

        batch_size = loc_out[0].size(0)
        loc_out = torch.cat([i.view(batch_size, -1, 4) for i in loc_out], 1) # shape [B, num_priors_per_img, 4]
        conf_out = torch.cat([i.view(batch_size, -1) for i in conf_out], 1)
        conf_out = conf_out.view(batch_size, loc_out.size(1), -1) # shape [B, num_priors_per_img, num_classes]
        
        loc_loss, conf_loss = 0., 0.
        for loc_out_i, conf_out_i, loc_target_i, conf_target_i in zip(loc_out, conf_out, loc_target, conf_target):
            loc_target_center_i = center_form(loc_target_i)
            pos_priors_mask_i, idx_obj_i = match(self.priors_point, loc_target_i)
            matched_prior = self.priors[pos_priors_mask_i]
            matched_target_i = loc_target_center_i[idx_obj_i]
            loc_regression_target_i = encode(matched_target_i, matched_prior, self.config)
            # loc loss
            loc_loss += F.smooth_l1_loss(loc_regression_target_i, loc_out_i[pos_priors_mask_i])

            num_pos = matched_prior.size(0)
            num_neg = num_pos * neg_rate
            conf_t = torch.zeros(conf_out_i.size(0)).type(torch.int64) # shape [num_priors_per_img]
            if self.config["cuda"]:
                conf_t = conf_t.cuda()
            conf_t[pos_priors_mask_i] = conf_target_i[idx_obj_i].reshape(-1)
            # print(conf_out_i.shape, "\n", conf_t[pos_priors_mask_i])

            # cls loss
            ce_loss = F.cross_entropy(conf_out_i, conf_t, reduction="none")
            neg_loss = ce_loss[~pos_priors_mask_i]
            neg_loss_sorted = torch.sort(neg_loss, descending=True)[0]
            num_neg = min(num_neg, neg_loss_sorted.size(0))
            conf_loss += (torch.sum(ce_loss[pos_priors_mask_i]) + torch.sum(neg_loss_sorted[0:num_neg])) / (num_neg + num_pos)
            # print(loc_loss, conf_loss)
        return loc_loss / batch_size, conf_loss / batch_size


if __name__ == "__main__":
    import numpy as np
    from net_model import SSD

    b, c, h, w = 1, 3, 300, 300
    num_classes = 2
    config["cuda"] = config["cuda"] and torch.cuda.is_available()

    ssd = SSD(num_classes)
    if config["cuda"]:
        ssd = ssd.cuda()
    x = torch.ones([b, c, h, w])
    predictions = ssd(x)

    mbox_loss = MultiBoxLoss()

    # targets = torch.Tensor([[0.4, 0.4, 0.6, 0.6, 4], [0.3, 0.3, 0.5, 0.5, 1], [-1, -1, -1, -1, -1]]).reshape(b, -1, 5)
    # loc_loss, conf_loss = mbox_loss(predictions, targets)
    # print(loc_loss, conf_loss)

    from torch.utils.data import DataLoader
    from data_set import detection_dataset, collate_fn
    data_file = "labels.txt"
    cls_file = "classes.txt"
    dataset = detection_dataset(data_file, cls_file)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch_data in loader:
        img_tensor, boxes_tensor = batch_data
        if config["cuda"]:
            img_tensor = img_tensor.cuda()
            boxes_tensor = boxes_tensor.cuda()
        predictions = ssd(img_tensor)
        loss = mbox_loss(predictions, boxes_tensor)
        print(loss)
        break
        
    
