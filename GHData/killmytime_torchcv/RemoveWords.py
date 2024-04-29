# coding=utf-8
# 图片修复

import cv2
import numpy as np

img = cv2.imread('test.jpg')
# blurred = cv2.GaussianBlur(img, (9, 9), 0)
# 图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
thresh = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))

# 创建形状和尺寸的结构元素
kernel = np.ones((1, 1), np.uint8)

# 扩张待修复区域
hi_mask = cv2.dilate(thresh, kernel, iterations=1)
specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_TELEA)
# gray = cv2.cvtColor(specular, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("Original_Image", img)
cv2.imshow("Specular", specular)
# cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
