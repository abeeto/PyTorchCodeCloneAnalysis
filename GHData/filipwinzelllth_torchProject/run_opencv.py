import cv2
import os

path = '/usr/matematik/fi5666wi/Documents/Datasets/Eval/cropped_image_slic.png'
image = cv2.imread(path)

print('Start')

WINDOW_NAME = 'Image de Prostata'
#cv2.namedWindow(WINDOW_NAME)
#cv2.startWindowThread()

cv2.imshow(WINDOW_NAME, image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()


print('Finished')
