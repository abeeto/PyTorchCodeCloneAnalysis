import torch

from core.resnet38_cls import Classifier

model = Classifier(20)

images = torch.randn([1, 3, 448, 448])
# images = torch.randn([1, 3, 224, 224])

logits = model(images)
print(logits.size())

logits, cams = model(images, with_cam=True)
print(logits.size(), cams.size())

# cams = model.forward_for_cam(images)
# print(cams.size()) # [1, 20, 56, 56]