import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.4060],
                         std=[0.229, 0.224, 0.225])
])

# print(dir(models))


# model = models.Inception3()
# alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True, progress=True)


# print(resnet)

image = Image.open('dog.jpg')
img_t = preprocess(image)
batch_t = torch.unsqueeze(img_t, 0)
print(batch_t.shape)
resnet.eval()

result = resnet(batch_t)
# print(f"result: {result}")
print(f"result: {np.argmax(result.detach().numpy())}")
print(f"result shape: {result.shape}")

with open('../dlwpt-code/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(result, 1)

_, sorted_index = torch.sort(result, descending=True)

# print(f"index: {index}")
percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100

# print(labels[index[0]], percentage[index[0]].item())


top_5 = [(labels[idx], percentage[idx].item()) for idx in sorted_index[0][:5]]

for item in top_5:
    print(item)

