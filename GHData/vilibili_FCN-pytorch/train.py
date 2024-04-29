import tqdm
import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.FCN_torch import FCN
from torch.utils.data import DataLoader
from utilities.dataReader import datareader

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# Hyper-parameters
input_size = 3
num_classes = 2
num_epochs = 25
batch_size = 2
learning_rate = 0.0001

image_list_path = r'dataset/train_list.txt'
image_test_path = r'dataset/images/image-name.jpeg'
mask_test_path = r'dataset/masks/image-name.png'
ckpt_path = r'ckpt\FCN_model.pytorch'
name = 'image-name'
image_test = imageio.imread(image_test_path)
imageio.imsave(os.path.join('dataset', name + '.jpeg'), image_test, format='jpeg')
image_test = torch.from_numpy(np.expand_dims(np.moveaxis(image_test, -1, 0), axis=0))
mask_test = imageio.imread(mask_test_path)
imageio.imsave(os.path.join('dataset', name + '.png'), mask_test, format='png')

model = FCN(num_classes=num_classes)

model.to(device)

print(model)
dtset = datareader(image_list_path)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate ,momentum=0.9, weight_decay=1/16)

for epoch in range(num_epochs):

    dt_loader_train = DataLoader(dtset, batch_size=batch_size, shuffle=True)
    for batch_index, batch in enumerate(dt_loader_train):

        image = batch[0]
        mask = batch[1]

        image = image.to(device)
        mask = mask.to(device=device, dtype=torch.long)
        y_pred = model(image)

        loss = criterion(y_pred, mask)
        print('Epoch ',epoch,'  iter ',batch_index*2,'  loss : ', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), ckpt_path)

    with torch.no_grad():
        image_test = image_test.to(device)

        # Generate prediction
        prediction = model(image_test)

        prediction = np.squeeze(prediction, axis=0)

        prediction = prediction.cpu().numpy()

        # Predicted class value using argmax
        predicted_class = np.argmax(prediction, axis=0)

        predicted_class = np.array(predicted_class, dtype=np.uint8)

        # Show result
        plt.imshow(predicted_class, cmap='gray')
        plt.show()
        predicted_class[predicted_class==1]=255

        imageio.imsave(os.path.join('dataset', name + '_predicted.png'), predicted_class, format='png')
