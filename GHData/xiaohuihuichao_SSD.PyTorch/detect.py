import torch

from config import config
from iou import jaccard_iou
from prior import get_prior
from box_transform import decode


class Detect:
    def __init__(self, config=config):
        self.config = config
    
    def __call__(self, predictions, threshold=0.5):
        return self.forward(predictions, threshold)

    def forward(self, predictions, threshold):
        # B = 1
        loc_out, conf_out = predictions
        
        batch_size = loc_out[0].size(0)
        feature_map_sizes = [l.shape[1:3] for l in loc_out]

        loc_out = torch.cat([i.view(batch_size, -1, 4) for i in loc_out], 1) # shape [B, num_priors_per_img, 4]
        conf_out = torch.cat([i.view(batch_size, -1) for i in conf_out], 1)
        conf_out = conf_out.view(batch_size, loc_out.size(1), -1) # shape [B, num_priors_per_img, num_classes]

        loc_out = loc_out[0] # shape [num_priors_per_img, 4]
        conf_out = conf_out[0] # shape [num_priors_per_img, num_classes]
        conf_out = torch.softmax(conf_out, dim=-1)

        priors = [get_prior(feature_map_size, self.config["image_size"], min_size, max_size, ratio) for feature_map_size, min_size, max_size, ratio in zip(feature_map_sizes, self.config["min_sizes"], self.config["max_sizes"], self.config["ratios"])]
        priors = torch.cat(priors, dim=0) # shape [num_priors_per_img, 4]
        if config["cuda"] and torch.cuda.is_available():
            priors = priors.cuda()
        loc_out = decode(loc_out, priors, self.config) # x1, y1, x2, y2
        
        # conf_out[:, 1:]  0是背景
        max_conf, max_idx = torch.max(conf_out[:, 1:], dim=1) # shape [num_priors_per_img], [num_priors_per_img]
        # obj_mask = torch.gt(max_conf, threshold).type(torch.bool)
        obj_mask = torch.gt(max_conf, threshold).type(torch.uint8) # shape [num_priors_per_img]

        obj_conf = max_conf[obj_mask] # shape [num_obj]
        obj_cls = max_idx[obj_mask] #  # shape [num_obj], 0是第一个obj类别
        obj_box = loc_out[obj_mask]  # shape [num_obj, 4]

        return obj_conf, obj_cls, obj_box


def nms(conf, box, iou_threshold=0.5):
    """最原始的 NMS \n
    Args:
        conf: shape [N]
        box: shape [N, 4]
    Return:
        keep_index: list
    """
    keep_index = []

    _, order_index = conf.sort(descending=True) # 降序

    while order_index.numel() > 0:
        keep_index.append(order_index[0].item())
        if order_index.numel() == 1:
            break

        ious = jaccard_iou(box[order_index[0:1]], box[order_index]).reshape(-1)
        k = ious <= iou_threshold
        order_index = order_index[k]
    return keep_index


if __name__ == "__main__":
    import os
    import cv2
    import collections
    from PIL import Image
    from torchvision.transforms import ToTensor

    from net_model import SSD

    img_path = "imgs/172_result.jpg"
    model_path = "model_file/epoch-50.pt"
    num_cls = 1
    threshold = 0.5
    iou_threshold = 0.5

    ssd = SSD(num_cls+1)
    if os.path.isfile(model_path):
        print("Loading model.")
        d = collections.OrderedDict()
        checkpoint = torch.load(model_path)
        for key, value in checkpoint.items():
            tmp = key[7:]
            d[tmp] = value
        ssd.load_state_dict(d)
    else:
        print(f"{model_path} 不存在")

    detect = Detect(config)

    img = Image.open(img_path)
    img = img.resize((300, 300))
    img = ToTensor()(img).unsqueeze(0)
    if torch.cuda.is_available() and config["cuda"]:
        img = img.cuda()
        ssd = ssd.cuda()
    
    with torch.no_grad():
        predictions = ssd(img)
        obj_confes, obj_clses, obj_boxes = detect(predictions, threshold)
        obj_boxes *= 300

    img_cv = cv2.imread(img_path)
    h, w = img_cv.shape[0:2]
    h /= 300
    w /= 300

    keep_index = nms(obj_confes, obj_boxes, iou_threshold)
    for idx in keep_index:
        box = obj_boxes[idx]
        conf = obj_confes[idx]
        c = obj_clses[idx]
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        cv2.rectangle(img_cv, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color=(0, 0, 255), thickness=2)
        cv2.putText(img_cv, f"{conf.item():.3f}", (int(x1*w), int(y1*h)), cv2.FONT_HERSHEY_PLAIN, 2, color=(0, 0, 255))
    cv2.imwrite(img_path, img_cv)
