import cv2
import numpy as np


image = cv2.imread(r'C:\Users\aero\PycharmProjects\opencv\hello\hello_2.jpg', 1)
# 二值化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
# cv2.imshow("cell", binary)
# cv2.waitKey(0)

rows, cols = binary.shape
scale = 20
# 识别横线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
# cv2.imshow("Eroded Image",eroded)
dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
cv2.imshow("Dilated Image", dilatedcol)

# 识别竖线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
cv2.imshow("Dilated Image", dilatedrow)

# 标识交点
bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
cv2.imshow("bitwiseAnd Image", bitwiseAnd)

# 标识表格
merge = cv2.add(dilatedcol, dilatedrow)
cv2.imshow("add Image", merge)
cv2.waitKey(0)
