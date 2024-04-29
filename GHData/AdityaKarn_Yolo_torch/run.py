from app import *
import cv2

img1 = './images/bar.jpeg'

cv2_img = cv2.imread(img1)

# cv2.imshow('image', cv2_img)

res = yolo_processing(cv2_img)

print(res)