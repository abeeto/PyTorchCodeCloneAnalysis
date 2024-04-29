import cv2

def selectiveSearch(img, numBoxes=2000, fast=True, show=False):
    im = cv2.imread(img)
    im = cv2.resize(im, (300, 300))

    search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    search.setBaseImage(im)
    if fast:
	    search.switchToSelectiveSearchFast()
    else:
	    search.switchToSelectiveSearchQuality()

    rects = search.process()

    if show:
        for count, (x, y, w, h) in enumerate(rects):
            if count < numBoxes:
                cv2.rectangle(im, (x, y), (x+w, y+h), (200, 255, 10), 1, cv2.LINE_AA)
            else:
                continue

        cv2.imshow("out", im)
        k = cv2.waitKey(0)

    return rects[:numBoxes]


def intersectionOverUnion(box1, box2):
    # computes how much two given reactangle in format (x, y, w, h) overlap
    # outputs intersected area devided by union area (between 0..1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def parseConfig(file):
    file = open(file, "r")
    lines = file.read().split("\n")
    lines = [i for i in lines if len(i) > 0 and i[0] != "#"] # np empty lines or comments
    lines = [i.strip() for i in lines]

    blocks = []

    for line in lines:
        if line[0] == '[':
            blocks.append({
                "name": line[1:-1].strip()
            })
        else:
            key, val = line.split('=')
            blocks[-1][key.strip()] = val.strip()

    return blocks       


blocks = parseConfig("yolov3.cfg")
print(blocks[0])