from __future__ import print_function
from __future__ import division
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet


file = "./test.png"

model = EfficientNet.from_pretrained("efficientnet-b7")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
trans = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img = trans(Image.open(file).convert("RGB"))
img = img.unsqueeze(0)
img = img.to(device)
with torch.no_grad():
    output = model(img)
    _, preds = torch.max(output, 1)
    print(f"Test passed, class number of test image image is {int(preds)}")
