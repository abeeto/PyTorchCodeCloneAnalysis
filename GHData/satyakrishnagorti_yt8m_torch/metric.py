import os
import numpy as np
import utils.mean_average_precision_calculator as map_calculator

class Labels(object):
  """Contains the class to hold label objects.
  This class can serialize and de-serialize the groundtruths.
  The ground truth is in a mapping from (segment_id, class_id) -> label_score.
  """

  def __init__(self, labels):
    """__init__ method."""
    self._labels = labels

  @property
  def labels(self):
    """Return the ground truth mapping. See class docstring for details."""
    return self._labels

  def to_file(self, file_name):
    """Materialize the GT mapping to file."""
    with open(file_name, "w") as fobj:
      for k, v in self._labels.items():
        seg_id, label = k
        line = "%s,%s,%s\n" % (seg_id, label, v)
        fobj.write(line)

  @classmethod
  def from_file(cls, file_name):
    """Read the GT mapping from cached file."""
    labels = {}
    with open(file_name) as fobj:
      for line in fobj:
        line = line.strip().strip("\n")
        seg_id, label, score = line.split(",")
        labels[(seg_id, int(label))] = float(score)
    return cls(labels)


def read_labels(cache_path=""):
  """Read labels from TFRecords.
  Args:
    data_pattern: the data pattern to the TFRecords.
    cache_path: the cache path for the label file.
  Returns:
    a Labels object.
  """
  if cache_path:
    if os.path.exists(cache_path):
      print("Reading cached labels from %s..." % cache_path)
      return Labels.from_file(cache_path)

  else:
      raise NotImplementedError("No cache file provided")


def read_segment_predictions(file_path, labels, top_n=None):
    """Read segement predictions.
    Args:
      file_path: the submission file path.
      labels: a Labels object containing the eval labels.
      top_n: the per-class class capping.
    Returns:
      a segment prediction list for each classes.
    """
    cls_preds = {}  # A label_id to pred list mapping.
    with open(file_path) as fobj:
        print("Reading predictions from %s..." % file_path)
        for line in fobj:
            line = line.strip()
            if line == "Class,Segments":  # get rid of the header
                continue
            label_id, pred_ids_val = line.split(",")
            pred_ids = pred_ids_val.split(" ")
            if top_n:
                pred_ids = pred_ids[:top_n]
            pred_ids = [
                pred_id for pred_id in pred_ids
                if (pred_id, int(label_id)) in labels.labels
            ]
            cls_preds[int(label_id)] = pred_ids
            if len(cls_preds) % 50 == 0:
                print("Processed %d classes..." % len(cls_preds))
        print("Finish reading predictions.")
    return cls_preds


def run_metric(submission_file, label_cache, top_n=100_000):
    eval_labels = read_labels(label_cache)
    print("Total rated segments: %d." % len(eval_labels.labels))
    positive_counter = {}
    negative_counter = {}
    for k, v in eval_labels.labels.items():
        _, label_id = k
        if v > 0:
            positive_counter[label_id] = positive_counter.get(label_id, 0) + 1
        else:
            negative_counter[label_id] = negative_counter.get(label_id, 0) + 1

    present_label_ids = sorted(list(positive_counter.keys()))
    present_label_ids_negatives = sorted(list(negative_counter.keys()))

    seg_preds = read_segment_predictions(submission_file, eval_labels, top_n=top_n)
    map_cal = map_calculator.MeanAveragePrecisionCalculator(len(seg_preds), present_label_ids=present_label_ids)
    seg_labels = []
    seg_scored_preds = []
    num_positives = []
    num_negatives = []

    index2class = {}
    count = 0
    for label_id in sorted(seg_preds):
        if label_id not in present_label_ids:
            continue
        class_preds = seg_preds[label_id]
        seg_label = [eval_labels.labels[(pred, label_id)] for pred in class_preds]
        seg_labels.append(seg_label)
        seg_scored_pred = []
        if class_preds:
            seg_scored_pred = [
                float(x) / len(class_preds) for x in range(len(class_preds), 0, -1)
            ]
        seg_scored_preds.append(seg_scored_pred)
        num_positives.append(positive_counter[label_id])
        index2class[count] = label_id
        count += 1

    map_cal.accumulate(seg_scored_preds, seg_labels, num_positives)
    map_at_n = np.mean(map_cal.peek_map_at_n())

    seg_labels_neg = []
    for label_id in sorted(seg_preds):
        if label_id not in present_label_ids_negatives:
            continue
        class_preds = seg_preds[label_id]
        seg_label = [eval_labels.labels[(pred, label_id)] for pred in class_preds]
        seg_labels_neg.append(seg_label)
        num_negatives.append(negative_counter[label_id])

    # additional metric calculations
    recall_ones = []
    for s, p in zip(seg_labels, num_positives):
        recall_ones.append(sum(s) / p)

    recall_ones_value = sum(recall_ones) / len(recall_ones)

    recall_zeros = []
    for s, n in zip(seg_labels_neg, num_negatives):
        recall_zeros.append((len(s) - sum(s)) / n)

    recall_zeros_value = sum(recall_zeros) / len(recall_zeros)

    print("mAP@{}:{}, recall_ones:{}, recall_zeros:{}".format(top_n, map_at_n, recall_ones_value, recall_zeros_value))

    return map_at_n, recall_ones_value, recall_zeros_value