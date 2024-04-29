import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import argparse

from utils import build_network

parser =argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net name')
parser.add_argument('-num_classes', type=int, default=5)
args = parser.parse_args()

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("../tulip.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = build_network(args)
# load model weights
model_weight_path = './' + args.net + '.pth'
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)  # strict 使得加载网络的时候严格按照现在构建的网络结构加载数据， 即当前不含两个aux分支, 加载的时候也不加载该分支
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)])
plt.show()