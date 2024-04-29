import torch
from torch.nn.utils.rnn import pack_padded_sequence


def sort_sequences(seqs, lengths, pack=True):
    length_sorted, sort_idx = lengths.sort(descending=True)
    _, unsort_idx = sort_idx.sort()
    zero_indices = torch.masked_select(
        torch.arange(0, length_sorted.shape[0], device=seqs.device),
        length_sorted.eq(0))
    if zero_indices.shape[0] > 0:
        zero_pos = zero_indices.min()
        length_sorted = length_sorted[:zero_pos]
        sort_idx = sort_idx[:zero_pos]
    seqs_sorted = seqs.index_select(0, sort_idx)
    if pack:
        packed_seqs = pack_padded_sequence(seqs_sorted, length_sorted, batch_first=True)
        return packed_seqs, unsort_idx
    else:
        return seqs_sorted, length_sorted, unsort_idx


def pad_timestamps_and_batches(seqs, original_shape):
    if seqs.shape[0] < original_shape[0] or seqs.shape[1] < original_shape[1]:
        seqs = torch.nn.functional.pad(
            seqs,
            [0, 0,
             0, original_shape[1] - seqs.shape[1],
             0, original_shape[0] - seqs.shape[0]
             ])
    return seqs


def unsort_sequences(seqs, unsort_idx, original_shape=None, unpack=True):
    if unpack:
        seqs, _ = torch.nn.utils.rnn.pad_packed_sequence(seqs, batch_first=True)
    if original_shape is not None:
        seqs = pad_timestamps_and_batches(seqs, original_shape)
    return seqs.index_select(0, unsort_idx)
