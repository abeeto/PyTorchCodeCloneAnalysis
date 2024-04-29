pkg_dir = 'model/pretrained/pretrained-models.pytorch-master/'
import sys
sys.path.insert(0, pkg_dir)
sys.path.append('../')
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F


from model import pretrained

class RetinEyeModel():
    def __init__(self):
        # self.model_path = m_path
        self.model = pretrained.__dict__['resnet101'](pretrained=None)

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Sequential(
                                  nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(in_features=2048, out_features=2048, bias=True),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features=2048, out_features=1, bias=True),
                                 )
        try:
            self.model.load_state_dict(torch.load("model.bin", map_location=torch.device('cpu')))
            for param in self.model.parameters():
                param.requires_grad = False
        except FileNotFoundError as e:
            print(e)


    def get_predictions(self, image_path):
        image = Image.open(image_path)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        image = transforms.ToTensor()(image)
        imgs = image.unsqueeze(0)
        self.model.eval()

        pred = self.model(imgs)[0]
        prob = F.softmax(pred, dim=-1)
        coef = [0.5, 1.5, 2.5, 3.5]

        if pred < coef[0]:
            test_preds = [0, 'No DR']
        elif pred >= coef[0] and pred < coef[1]:
            test_preds = [1, 'Mild']
        elif pred >= coef[1] and pred < coef[2]:
            test_preds = [2, 'Moderate']
        elif pred >= coef[2] and pred < coef[3]:
            test_preds = [3, 'Severe']
        else:
            test_preds = [4, 'Proliferative DR']

        return test_preds, prob

if '__main__' == __name__:
    import os
    obj = RetinEyeModel()
    imgs = os.listdir('test_images')
    for img in imgs:
        img_path = 'test_images/' + img
        print(img, obj.get_predictions(img_path)[0])