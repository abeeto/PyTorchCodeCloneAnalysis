import cv2
import numpy


def get_red(img):
    reaImg = img[:, :, 2]
    return reaImg


def get_green(img):
    return img[:,:,1]


def get_blue(img):
    return img[:, :, 0]

if __name__ == '__main__':
    # img = cv2.imread("gray.jpg")
    # img = cv2.imread('luis.jpg')
    img = cv2.imread('1_2.jpg')
    #print(img)
    cv2.imshow('img',img)
    b,g,r = cv2.split(img)
    print(b)
    print(g)
    print(r)
    cv2.imshow("b", b)
    # cv2.imshow("g", g)
    # cv2.imshow("r", r)


    # b1 = get_blue(img)
    print(img.shape)
    img = cv2.resize(img,(32,32))
    print(img.shape)
    cv2.imshow('b2',get_blue(img))
    cv2.imshow('r2',get_red(img))
    gray = (get_red(img)*30+get_green(img)*59+get_blue(img)*11)/100
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()