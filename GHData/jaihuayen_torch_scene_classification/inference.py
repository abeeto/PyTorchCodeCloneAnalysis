import torch
import torch.nn as nn
import os
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from dataset import get_dataset

train_dataset, val_dataset, test_dataset = get_dataset(IMG_SIZE=224)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./models/model_mobilenetv2.pt')

img_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model.eval()

dir = '/home/jhy/Desktop/video_scene/data/frames/How To Organize Paperwork And Files'

img_list = os.listdir(dir)

class_names = train_dataset.classes
post_mapping = {i: name for i, name in enumerate(class_names)}
softmax = nn.Softmax(dim=1)

mapping = dict()

for _, img in enumerate(img_list):
    image = Image.open(os.path.join(dir, img))
    t_image = img_transforms(image)
    s_image = torch.unsqueeze(t_image, 0).cuda()
    output = model(s_image)
    soft_output = softmax(output)
    pred = soft_output.max(1, keepdim=True)[1].detach().cpu().numpy()[0][0]
    pred_class = [post_mapping[pred], soft_output.detach().cpu().numpy().max(1)[0]]
    mapping[img] = pred_class

mapping_result = pd.DataFrame(list(mapping.items()), columns=['jpg', 'class'])
mapping_result[['category', 'prob']] = pd.DataFrame(mapping_result['class'].tolist(), index=mapping_result.index)

mapping_result = mapping_result.drop('class', 1)

mapping_result.loc[mapping_result.prob < 0.5, 'category'] = ''

mapping_result = mapping_result.sort_values('jpg').reset_index(drop=True)

print(mapping_result)
mapping_result.to_csv('image_class.csv')