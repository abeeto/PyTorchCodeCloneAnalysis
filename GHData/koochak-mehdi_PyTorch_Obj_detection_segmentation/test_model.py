import torch as t
import torchvision as tv
from main import get_transform
from main import get_model_instance_segmentation

import cv2
import numpy as np
from PIL import Image

FILE = 'trainedModel.pth'

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

num_classes = 2
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(t.load(FILE))
model.eval()

T_to_PIL = tv.transforms.ToPILImage()
PIL_to_T = tv.transforms.ToTensor()

pil_img = Image.open(r'data\test\person_walking.jfif')
cv2_img = np.array(pil_img)[:, :, ::-1].copy()

prediction = model([PIL_to_T(cv2_img)])[0]

scores = prediction['scores'] >= .8
boxes = prediction['boxes']

for box in boxes[:np.count_nonzero(scores)]:
    box = (box.detach().numpy()).astype(np.int64)
    #print(box)
    cv2.rectangle(cv2_img,
                (box[0], box[1]),
                (box[2], box[3]),
                (255, 0, 0),
                2)
    
cv2.imshow('show', cv2_img)
cv2.waitKey()
cv2.destroyAllWindows()