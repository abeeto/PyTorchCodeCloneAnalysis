import torch
import numpy as np


def cross_entropy_loss(predictions, labels, label_weights=None):
    """
    Multi label BCE
    :param predictions: (B, 3862)
    :param labels: (B, 3862)
    :param label_weights:
    :return: loss
    """

    epsilon = 1e-5
    cross_entropy_loss = -(labels * torch.log(predictions + epsilon) +
                           (1-labels) * torch.log(1 - predictions + epsilon))
    if label_weights is not None:
        cross_entropy_loss *= label_weights
    return torch.mean(torch.sum(cross_entropy_loss, 1))


def masked_cross_entropy(predictions, labels, label_weights, label_masks):
    epsilon = 1e-5
    cross_entropy_loss = (labels * label_weights) * torch.log(predictions + epsilon) + \
                         ((1 - labels) * label_weights) * torch.log(1 - predictions + epsilon)

    cross_entropy_loss = - cross_entropy_loss
    # video_loss: (B, 1)
    video_loss = torch.sum(torch.sum(cross_entropy_loss, 1), 1)

    # num of labelled segments in each video: (B, 1)
    num_labelled_segs = torch.sum(torch.sum(label_weights, 1), 1)

    segment_loss = video_loss / num_labelled_segs

    return torch.mean(segment_loss)