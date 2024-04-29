import torch
import numpy as np
import time
import torchvision

model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_1', pretrained=True)
model.eval()

import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "cat.png")

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

torch.set_num_threads(4)

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

with torch.no_grad():
    out = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

repeat=10
numpy_time = np.zeros(repeat)
for i in range(0,repeat):
    start_time = time.time()
    with torch.no_grad():
        out = model(input_batch)

    elapsed_ms = (time.time() - start_time) * 1000
    numpy_time[i] = elapsed_ms

print("pytorch Squeezenet v1.1  %-19s (%s)" % ("%.2f ms" % np.mean(numpy_time), "%.2f ms" % np.std(numpy_time)))

#_, index = torch.max(out, 1)
#percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

#with open('mobilenet-v2-labels.txt') as f:
#   labels = [line.strip() for line in f.readlines()]

#_, indices = torch.sort(out, descending=True)
#percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#[print(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

