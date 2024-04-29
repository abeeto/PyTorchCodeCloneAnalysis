from yolo.yolov5 import Yolo
import torch
import yolo.tool as tool
import torch.optim as optim

import cv2 as cv

device = 'cuda'
yolo = Yolo().to(device=device)

train_x, train_y, label = tool.load_dataset(yolo, './dataset', 'cpu')

optimizer = optim.Adam(yolo.parameters(), lr=1e-4)
# tool.load_model_train(yolo, optimizer, 'cs_model_1.pt')
tool.train(yolo, train_x, train_y,
           batch_size=8,
           epoch=1000,
           optimizer=optimizer,
           device='cuda')
tool.save_model(yolo, optimizer, 'cs_model_1.pt')
img, x = tool.open_img('csgo659.jpg', 'cuda')
pred = tool.detect(yolo, x)
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
for bs in pred:
    for border in bs:
        cv.rectangle(img, border[1], border[2], (0, 0, 255), 1)
cv.namedWindow('main')
cv.imshow('main', img)
cv.waitKey()
