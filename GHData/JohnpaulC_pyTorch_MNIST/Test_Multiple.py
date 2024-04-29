import torch
import cv2 as cv
from model import ConvNN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNN()
model.load_state_dict(torch.load('model.pt', map_location=device))

img = cv.imread('number_matrix.bmp')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

img_size = 28
_, h, w = img.shape
index_h = int(h / img_size) - 1
index_w = int(w / img_size) - 1
final_sum = 0


for i in range(index_h):
    for j in range(index_w):
        image = img[:, i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size]
        image = image.unsqueeze(0)
        _, pred = model(image, apply_softmax=True).max(dim=1)
        final_sum += pred.item()

        if False:
            print("The prediction is {0:d}".format(pred.item()))
            image_show = image.squeeze().numpy()
            cv.imshow('Image', image_show)
            cv.waitKey()
            cv.destroyWindow('fuck')

print(final_sum)