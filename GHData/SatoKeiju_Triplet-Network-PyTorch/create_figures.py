import cv2
import numpy as np


print(f'cv2: version {cv2.__version__}')


def draw_circle(img_size, file_type='jpg'):
    img = np.full(img_size, 255, dtype=np.uint8)

    cv2.circle(
        img,
        center=(32, 32),
        radius=16,
        color=(0, 0, 0),
        thickness=-1
    )

    cv2.imwrite('data/circle.'+file_type, img)


def draw_triangle(img_size, file_type='jpg'):
    img = np.full(img_size, 255, dtype=np.uint8)

    cv2.fillPoly(
        img,
        pts=[np.array(((32, 16), (16, 48), (48, 48)))],
        color=(0, 0, 0)
    )

    cv2.imwrite('data/triangle.'+file_type, img)


def draw_rectangle(img_size, file_type='jpg'):
    img = np.full(img_size, 255, dtype=np.uint8)

    cv2.rectangle(
        img,
        pt1=(16, 16),
        pt2=(48, 48),
        color=(0, 0, 0),
        thickness=-1
    )

    cv2.imwrite('data/rectangle.'+file_type, img)

img_size = (64, 64, 3)

draw_circle(img_size)
draw_triangle(img_size)
draw_rectangle(img_size)
