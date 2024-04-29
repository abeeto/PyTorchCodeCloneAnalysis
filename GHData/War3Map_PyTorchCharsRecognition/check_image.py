import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from functools import cmp_to_key


def cv_show_image(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


TEST_IMAGE_PATH = r"C:\Users\IVAN\Desktop\PythonProgs\test_data\TestFonts.png"
TEST_IMAGE_PATH = r"C:\Users\IVAN\Desktop\PythonProgs\test_data\SimpleFont.png"
image = cv2.imread(TEST_IMAGE_PATH)

# Create a black image with same dimensions as our loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))
# Create a copy of our original image
orginal_image = image

# Grayscale our image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)
# cv2.imshow('1 - Canny Edges', edged)
# cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_NONE)[:2]


def show_detected(images_list):
    """
    Shows images in one grid
    :param images_list: list of images
    :return:
    """
    images_per_row = 20
    images_count = len(images_list)
    images_per_column = images_count // images_per_row + 1

    fig = plt.figure(figsize=(1000, 1000))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(images_per_column, 20),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images_list):
        # Iterating over the grid returns the Axes.
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.imshow(im)

    plt.axis('off')
    # plt.savefig("test.png", bbox_inches='tight')
    plt.show()


# plt.subplot(121)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(122)
# plt.imshow(edged, cmap='gray')
# plt.title('Edge Image')
# plt.xticks([])
# plt.yticks([])
# plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# thresh_val = gray.max()
# thresh_val = thresh_val/2 + 60

thresh_val = gray.mean()
thresh_val = thresh_val - 20
thresh_val = 180

ret, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
# img_erode = cv2.erode(cur_image, np.ones((3, 3), np.uint8), iterations=10)
# img_erode = cv2.dilate(cur_image, np.ones((3, 3), np.uint8), iterations=10)
# cv2.imshow(f"Source", image)

output = image.copy()
detected_objects = []
correct_contours = []
MIN_SQR = 40


for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent

    # sqr = w * h
    # if hierarchy[0][idx][3] == 0:
    #     if sqr > MIN_SQR:
    #         cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
    #
    #         cur_image = image[y:y + h, x:x + w]
    #         detected_objects.append(cur_image)
    if hierarchy[0][idx][3] == 0:
        sqr = w * h
        if sqr > MIN_SQR:
            cv2.rectangle(img_erode, (x, y), (x + w, y + h), (70, 0, 0), 1)

            cur_image = image[y:y + h, x:x + w]
            detected_objects.append(cur_image)
            correct_contours.append(contour)
    # cv2.imshow(f"Contour{idx}", cur_image)
    # input()

#sort contours
# sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

# sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
# # x + y * w


def contour_sort(a, b):

    br_a = cv2.boundingRect(a)
    br_b = cv2.boundingRect(b)

    if abs(br_a[1] - br_b[1]) <= 15:
        return br_a[0] - br_b[0]

    return br_a[1] - br_b[1]


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


# sorted_ctrs = sorted(correct_contours, key=lambda x:get_contour_precedence(x, thresh.shape[1]))
sorted_ctrs = sorted(correct_contours, key=cmp_to_key(contour_sort))

shutil.rmtree(r'.\\contours', ignore_errors=True)
os.mkdir('contours')

sorted_images = []
for idx, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    cur_image = image[y:y + h, x:x + w]
    sorted_images.append(cur_image)
    cv2.imwrite(f"contours\\contour{idx}.png", cur_image)

# draw contours image
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# cv2.waitKey(0)
cv_show_image(image, "image")
# cv_show_image(gray, "gray")
cv_show_image(img_erode, "thresh_erode")
# cv_show_image(thresh, "thresh")
# cv_show_image(edged, "edged")
cv_show_image(output, "result")

# cv_show_image(cont_image, "contours")
# show_images(detected_objects)

# show_grid(detected_objects)

show_detected(sorted_images)

