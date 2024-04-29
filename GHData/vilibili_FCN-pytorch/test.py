import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from model.FCN_torch import FCN

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

num_classes = 2
image_test_path = r'dataset/images/image-name.jpeg'
mask_test_path = r'dataset\masks\image-name.png'
ckpt_path = r'ckpt\FCN_model.pytorch'
name = 'image-name'
image_test = imageio.imread(image_test_path)
imageio.imsave(os.path.join('dataset', name + '.jpeg'), image_test, format='jpeg')
image_test = torch.from_numpy(np.expand_dims(np.moveaxis(image_test, -1, 0), axis=0))
mask_test = imageio.imread(mask_test_path)
imageio.imsave(os.path.join('dataset', name + '.png'), mask_test, format='png')

model = FCN(num_classes=num_classes)

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print('the checkpoint is loaded successfully.')
else:
    print('the checkpoint is not found.')
    exit()

model.eval()

model.to(device)

with torch.no_grad():
    image_test = image_test.to(device)

    # Generate prediction
    prediction = model(image_test)

    prediction = np.squeeze(prediction, axis=0)

    prediction = prediction.cpu().numpy()

    # Predicted class value using argmax
    predicted_class = np.argmax(prediction, axis=0)

    predicted_class = np.array(predicted_class, dtype=np.uint8)

    predicted_class[predicted_class == 1] = 255

    # Show result
    plt.imshow(predicted_class, cmap='gray')
    plt.show()
    
    predicted_class[predicted_class==1]=255

    imageio.imsave(os.path.join('dataset', name + '_predicted.png'), predicted_class, format='png')
