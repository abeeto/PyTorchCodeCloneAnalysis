import os
import pathlib

import numpy as np
import torch
from cv2 import cv2
from torch.utils.data.dataset import Dataset


class BmiDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.x_paths = pathlib.Path(path).glob('X_*.npy')
        self.x_paths = sorted([x for x in self.x_paths])
        self.bmi_paths = pathlib.Path(path).glob('y_bmi_*.npy')
        self.bmi_paths = sorted([x for x in self.bmi_paths])

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, index):
        x_path = str(self.x_paths[index])
        bmi_path = str(self.bmi_paths[index])

        X = torch.from_numpy(np.load(x_path, allow_pickle=True))
        bmi = torch.from_numpy(np.load(bmi_path, allow_pickle=True))

        return X, bmi


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]


def extract_face(image_path, face_detector, detector_input_name):
    orig_image = cv2.imread(str(image_path))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    confidences, boxes = face_detector.run(None, {detector_input_name: image})
    boxes, _, _ = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.7)

    if len(boxes) == 0:
        return None

    box = boxes[0]
    out_size = 112
    img = orig_image.copy()
    height, width, _ = img.shape
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h]) * 1.1)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    new_bbox = BBox(new_bbox)
    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        return None

    cropped_face = np.asarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    cropped_face = np.transpose(cropped_face, [2, 0, 1])
    return cropped_face


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def write_batch(batch_num, faces, bmis, out_path):
    # Normalize pixels from 0 to 1
    face_batch = np.asarray(faces, dtype='float32') / 255.0
    face_batch = face_batch.reshape(face_batch.shape[0], 3 * 112 * 112)
    bmi_batch = np.asarray(bmis, dtype='float32')
    bmi_batch = bmi_batch.reshape(bmi_batch.size, 1)

    face_file = os.path.join(out_path, 'X_%03d.npy' % (batch_num,))
    bmi_file = os.path.join(out_path, 'y_bmi_%03d.npy' % (batch_num,))

    np.save(face_file, face_batch)
    np.save(bmi_file, bmi_batch)


def extract_labels(image_path):
    parts = str(image_path).split('-')
    height = float(parts[3])
    weight = float(parts[4])
    bmi = weight / (height ** 2)

    return bmi


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    top_lefts0 = boxes0[..., :2]
    top_lefts1 = boxes1[..., :2]
    overlap_left_top = np.maximum(top_lefts0, top_lefts1)
    bottom_rights0 = boxes0[..., 2:]
    bottom_rights1 = boxes1[..., 2:]
    overlap_right_bottom = np.minimum(bottom_rights0, bottom_rights1)

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]
