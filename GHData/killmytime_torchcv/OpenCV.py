import cv2
import numpy as np


def get_image(path):
    # 获取图片
    img = cv2.imread(path)
    # img = cv2.resize(img, (800, 1000))
    # 转换灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def Sobel_gradient(gray):
    # 索比尔算子来计算x、y方向梯度，综合得到模糊边界
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient


def Thresh_and_blur(gradient):
    # 设定阈值
    blurred = cv2.GaussianBlur(gradient, (11, 11), 0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    return thresh


def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed


def findcnts_and_box_point(closed):
    # Todo 识别图片区域emmm，想没到
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
    print(len(cnts))
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]

    return draw_img, crop_img


def run():
    # 貌似在IDE里面，用绝对路径才行，换用IDE是因为，，还是有代码补全舒服
    # 具体原因以后再看
    img_path = 'paper-png/P19-1001/2.png'
    save_path = 'test0.png'
    original_img, gray = get_image(img_path)
    gradX, gradY, gradient = Sobel_gradient(gray)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, cut_img = drawcnts_and_cut(original_img, box)

    # 暴力一点，把它们都显示出来看看

    cv2.imshow('original_img', original_img)
    # cv2.imshow('gray', gray)
    # cv2.imshow('gradX', gradX)
    # cv2.imshow('gradY', gradY)
    # cv2.imshow('gradient', gradient)
    cv2.imshow('thresh', thresh)
    # cv2.imshow('closed', closed)
    cv2.imshow('draw_img', draw_img)
    # cv2.imshow('cut_img', cut_img)
    cv2.waitKey(20190909)
    # cv2.imwrite(save_path, crop_img)
    # cv2.destroyAllWindows()


run()
