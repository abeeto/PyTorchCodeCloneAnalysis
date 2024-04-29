import glob
from xml.etree import ElementTree as ET

import numpy as np

SIZE = 608
CLUSTERS = 9
ANNOTATIONS_PATH = "/home/jianghuixin/Datasets/VOCdevkit/VOC2007/Annotations"


def iou(box, clusters):
    """
    计算目标框与每一个族中心的 IoU
    其中 x = width, y = height
    :param box:
    :param clusters:
    :return:
    """
    x = np.minimum(box[0], clusters[:, 0])
    y = np.minimum(box[1], clusters[:, 1])

    if np.count_nonzero(x <= 0) or np.count_nonzero(y <= 0):
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    return intersection / (box_area + cluster_area - intersection)


def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    k-means 聚类, 使用 IoU 衡量距离
    :param boxes: 聚类输入数据
    :param k: 聚类中心个数
    :param dist:
    :return:
    """
    num = boxes.shape[0]

    distances = np.empty(shape=(num, k))
    # 每一个原始数据所属
    last_clusters = np.zeros(num)

    # 起始随机聚类中心
    clusters = boxes[np.random.choice(num, size=k, replace=False)]

    while True:
        for idx in range(num):
            distances[idx] = iou(boxes[idx], clusters)

        nearest_clusters = np.argmax(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # 重新选择聚类中心
        for cluster_idx in range(k):
            clusters[cluster_idx] = dist(boxes[nearest_clusters == cluster_idx], axis=0)

        last_clusters = nearest_clusters

    return clusters


def load_dataset(path):
    """
    使用 VOC 数据集测试 kmeans
    :param path: .../VOCdevkit/VOC2007/Annotations
    :return: N*2 ndarray
    """
    dataset = []
    for xml_file in glob.glob(f"{path}/*xml"):
        tree = ET.parse(xml_file)

        width = int(tree.findtext("./size/width"))
        height = int(tree.findtext("./size/height"))

        for obj in tree.iter("object"):
            x1 = int(obj.findtext("bndbox/xmin")) / width
            y1 = int(obj.findtext("bndbox/ymin")) / height
            x2 = int(obj.findtext("bndbox/xmax")) / width
            y2 = int(obj.findtext("bndbox/ymax")) / height

            dataset.append([x2 - x1, y2 - y1])

    return np.array(dataset)


if __name__ == "__main__":
    anno_box = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(anno_box, CLUSTERS)
    # 按宽度排序
    anchor_box = out[np.argsort(out[:, 0])]
    ANCHORS = (anchor_box * SIZE).ravel()
    print("ANCHORS:", ANCHORS, sep="\n")
    print(f"Accuracy: {avg_iou(anno_box, out) * 100:.2f}")
