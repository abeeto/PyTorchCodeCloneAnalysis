import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

MAX_BRIGHTNESS = 255
COLOR_BLUE = (MAX_BRIGHTNESS, 0, 0)
COLOR_RED = (0, 0, MAX_BRIGHTNESS)
COLOR_YELLOW = (0, MAX_BRIGHTNESS, MAX_BRIGHTNESS)
COLOR_GREEN = (0, MAX_BRIGHTNESS, 0)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def draw_rect(img, points, color=COLOR_GREEN):
    # obtáhnout parkovací místo
    cv2.line(img, points[0], points[1], color, 3)
    cv2.line(img, points[1], points[2], color, 3)
    cv2.line(img, points[2], points[3], color, 3)
    cv2.line(img, points[3], points[0], color, 3)

def draw_dotted_rect(img, points, color=COLOR_GREEN):
    # obtáhnout parkovací místo
    draw_line(img, points[0], points[1], color, 3, style='dotted', gap=20)
    draw_line(img, points[1], points[2], color, 3, style='dotted', gap=20)
    draw_line(img, points[2], points[3], color, 3, style='dotted', gap=20)
    draw_line(img, points[3], points[0], color, 3, style='dotted', gap=20)

def draw_dotted_cross(img, points, color=COLOR_GREEN):
    # obtáhnout parkovací místo
    draw_line(img, points[0], points[2], color, 3, style='dotted', gap=20)
    draw_line(img, points[3], points[1], color, 3, style='dotted', gap=20)


def draw_cross(img, points, color=COLOR_RED):
    # křížek na body parkovacího místa
    cv2.line(img, points[0], points[2], color, 3)
    cv2.line(img, points[3], points[1], color, 3)


def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def get_true_results(filename='groundtruth.txt'):
    with open(filename) as truth_file:
        truth = [int(x) for x in truth_file.read().splitlines()]
    return truth


def get_points_float(one_c):
    # načtu si body parkovacích míst ve floatu
    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]
    return pts


def get_points_int(coordinate):
    # načtu si body parkovacích míst v int
    point_1 = (int(coordinate[0]), int(coordinate[1]))
    point_2 = (int(coordinate[2]), int(coordinate[3]))
    point_3 = (int(coordinate[4]), int(coordinate[5]))
    point_4 = (int(coordinate[6]), int(coordinate[7]))
    return [point_1, point_2, point_3, point_4]


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def get_coordinates(filename='parking_map_python.txt'):
    # načtu souřadnice bodů - čtveřice pro každé parkovací místo
    pkm_file = open(filename, 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    return pkm_coordinates


def get_parking_evaluation(TP, TN, FP, FN, i):
    # vyhodnotím všechny parkovací místa z obrázku
    precision = float(float(TP) / float(TP + FP))
    sensitivity = float(float(TP) / float(TP + FN))
    F1 = 2.0 * (float(precision * sensitivity) / float(precision + sensitivity))
    mcc_sqrt = math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
    MCC = float(TP * TN - FP * FN) / float(mcc_sqrt)
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "sensitivity": sensitivity,
        # f1 score - harmonic mean of precision and sensitivity
        "f1": F1,
        "accuracy": (float)(TP + TN) / (float)(i),
        "mcc": MCC
    }


def print_evaluation_header():
    print("\tTP\tTN\tFP\tFN\tprecision\tsensitivity\tf1\t\taccuracy\tMCC")


def print_evaluation_result(result):
    print("\t{:.0f}".format(result.get("TP")), end="\t")
    print("{:.0f}".format(result.get("TN")), end="\t")
    print("{:.0f}".format(result.get("FP")), end="\t")
    print("{:.0f}".format(result.get("FN")), end="\t")
    print("{:.4f}".format(result.get("precision")), end="\t\t")
    print("{:.4f}".format(result.get("sensitivity")), end="\t\t")
    print("{:.4f}".format(result.get("f1")), end="\t\t")
    print("{:.4f}".format(result.get("accuracy")), end="\t\t")
    print("{:.4f}".format(result.get("mcc")))
