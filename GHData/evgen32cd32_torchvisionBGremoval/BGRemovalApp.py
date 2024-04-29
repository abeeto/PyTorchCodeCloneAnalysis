import sys;
import os;
from PIL import Image;
import torch;
import torchvision.transforms as transforms;
import torchvision;
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor;
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor;


def removeBG(filepath):
    num_classes = 2;
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True);
    in_features = model.roi_heads.box_predictor.cls_score.in_features;
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes);
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels;
    hidden_layer = 256;
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes);
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
    model = model.to(device);
    model.load_state_dict(torch.load('./maskrcnn_resnet50_fpn_8',map_location=torch.device(device)));
    model = model.eval();
    
    img = transforms.ToTensor()(Image.open(filepath));
    img = img.to(device);
    output = model(torch.unsqueeze(img,dim=0));
    mask = (output[0]['masks'][output[0]['scores'].argmax()] >= 0.5).float()[0];
    oimg = transforms.ToPILImage()(img[:] * mask);
    pre, ext = os.path.splitext(filepath);
    oimg.save(pre + '_bgr' + ext, "PNG");

if __name__ == "__main__":
    removeBG(sys.argv[1]);
