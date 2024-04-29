import cv2
import imutils
import numpy
import time
import ssl
import logging
import os


logger = logging.getLogger(__name__)
ssl._create_default_https_context = ssl._create_unverified_context


def get_image(uri):
    """
    get image by local uri or remote url
    Parameters
    ----------
    uri: str
        local uri or remote url
    Returns
    -------
    asarray
        image data
    """
    if os.path.exists(uri):
        return cv2.imread(uri, cv2.IMREAD_UNCHANGED)
    else:
        return imutils.url_to_image(uri, cv2.IMREAD_UNCHANGED)


def crop(image, offset = 10):
    """
    crop an image to remove void space
    Parameters
    ----------
    image: asarray
        image data
    offset: int
        returning image offset, default = 10
    Returns
    ----------
    asarray
        image data
    """
    start = time.time()
    grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_ret, gray_threshold = cv2.threshold(grayed, 127, 255, 0)
    gray_contours_ret, gray_contours, gray_hierarchy = cv2.findContours(gray_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(grayed, gray_contours, -1, (0, 255, 0), 3)
    gray_mask = numpy.zeros_like(grayed)
    cv2.drawContours(gray_mask, gray_contours, -1, 255, -1)
    output = numpy.zeros_like(grayed)
    output[gray_mask == 255] = grayed[gray_mask == 255]
    (x, y) = numpy.where(gray_mask == 255)
    (top_x, top_y) = (numpy.min(x), numpy.min(y))
    (bottom_x, bottom_y) = (numpy.max(x), numpy.max(y))
    output = image[top_x : bottom_x + 1 : , top_y : bottom_y + 1 : ]
    output = cv2.copyMakeBorder(output, top = offset, bottom = offset, left = offset, right = offset, borderType = cv2.BORDER_CONSTANT, value = cv2.BORDER_DEFAULT)
    end = time.time()
    logger.info(f"crop timing: {end - start}")
    return output


def overlay_images(back, fore, x, y):
    """
    overlay a image on another
    Parameters
    ----------
    back: asarray
        background image
    fore: asarray
        foreground image
    x: int
        horizonal offset
    y: int
        vertical offset
    Returns
    ----------
    None
    """
    fore = cv2.cvtColor(fore, cv2.COLOR_BGR2BGRA)
    rows, cols, channels = fore.shape    
    trans_indices = fore[...,3] != 0 # Where not transparent
    overlay_copy = back[y:y+rows, x:x+cols] 
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy


def style_transfer(input, net, subtraction):
    """
    style transfer based on dnn model
    Parameters
    ----------
    input: asarray
        source image
    net: net
        neural network model
    subtraction: turple
    Returns
    ----------
    asarray
        transfered image
    """
    start = time.time()
    (height, width) = input.shape[:2]
    blob = cv2.dnn.blobFromImage(input, 1, (width, height), subtraction, swapRB = False, crop = False)
    net.setInput(blob)
    transfer_ret = net.forward()
    transfer_ret = transfer_ret.reshape((3, transfer_ret.shape[2], transfer_ret.shape[3]))
    transfer_ret[0] += subtraction[0]
    transfer_ret[1] += subtraction[1]
    transfer_ret[2] += subtraction[2]
    transfer_ret = transfer_ret.transpose(1, 2, 0)
    end = time.time()
    logger.info(f"transfer timing: {end - start}")
    return transfer_ret