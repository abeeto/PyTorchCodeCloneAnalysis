"""
Utilities for evaluation

author: Danfei Xu
"""
import numpy as np


def seq_accuracy(preds, labels, seq_len):
    """ Computes sequence classification accuracy
    preds:   [batch, time, n_cls]
    labels:  [batch, time]
    seq_len: [batch]
    """
    preds_label = np.argmax(preds, axis=2)
    num_correct = 0
    for pred, al, l in zip(preds_label, labels, seq_len):
        num_correct += np.sum(pred[:l] == al[:l])
    accuracy = float(num_correct) / float(np.sum(seq_len))
    return accuracy
