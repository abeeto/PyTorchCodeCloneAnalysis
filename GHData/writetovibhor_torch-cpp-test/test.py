from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import glob
import os
import numpy as np
import cv2

ksize = (15, 15)
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()

for filename in glob.glob('input/*.jpg'):
    if os.path.exists(f'output/{os.path.basename(filename)}'):
        continue
    img = read_image(filename)
    batch = preprocess(img).unsqueeze(0)
    print(f'batch = {batch.shape}')
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["person"]].detach().numpy()
    mask -= np.min(mask)
    mask = mask / np.max(mask)
    mask *= 255.0
    img = img.detach().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blurred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    bg = 255.0 - mask
    blurred = (blurred * bg) / 255.0
    img = (img * mask) / 255.0
    img = img + blurred
    cv2.imwrite(f'output/{os.path.basename(filename)}', img)
