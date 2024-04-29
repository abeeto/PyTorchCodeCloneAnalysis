import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Hemanvi.jpeg")

img = img[50:250, 40:240, :]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow("img_gray", img_gray)
# print(img.shape)


# img_gray_small = cv2.resize(img_gray, (25, 25))
# cv2.imshow("img_gray_small", img_gray_small)


# print(img_gray)
# print(img_gray_small)
#
crop = img[-3:, -3:]
print(crop)
#
# cv2.imshow("crop", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()


