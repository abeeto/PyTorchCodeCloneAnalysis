"""
Common PyTorch Utilities

author: Danfei Xu
"""

import torch
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from builtins import range

USE_GPU = torch.cuda.is_available()


def init_weight(weight, method='xavier_normal', gain=np.sqrt(2)):
    """Initialize weights with methods provided by nn.init (in place)

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    if method == 'xavier_normal':
        init.xavier_normal(weight, gain=gain)
    elif method == 'xavier_uniform':
        init.xavier_uniform(weight, gain=gain)
    elif method == 'orthogonal':
        init.orthogonal(weight, gain=gain)
    elif method == 'uniform':
        init.uniform(weight)
    elif method == 'normal':
        init.normal(weight)
    else:
        raise NotImplementedError('init method %s is not implemented' % method)


def init_fc(fc, method='xavier_normal', gain=np.sqrt(2), has_bias=True):
    """ Initialize a fully connected layer

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    init_weight(fc.weight, method, gain)
    if has_bias:
        init.constant(fc.bias, 0)
    else:
        print('NOT initializing bias for a layer!')


def init_rnn(rnn, method='xavier_normal', gain=np.sqrt(2)):
    """ Initialize a multi-layer RNN

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    for layer in range(rnn.num_layers):
        init_rnn_cell(rnn, method, gain, layerfix='_l%i' % layer)


def init_rnn_cell(rnn, method='xavier_normal', gain=np.sqrt(2), layerfix=''):
    """ Initialize an RNN cell (layer)

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
        layerfix: postfix of the layer name
    """
    init_weight(getattr(rnn, 'weight_ih' + layerfix), method, gain)
    init_weight(getattr(rnn, 'weight_ih' + layerfix), method, gain)
    init.constant(getattr(rnn, 'bias_ih' + layerfix), 0)
    init.constant(getattr(rnn, 'bias_hh' + layerfix), 0)


def to_tensor(np_array, cuda=True):
    """ Convert a numpy array to a tensor
    """
    if USE_GPU and cuda:
        return torch.from_numpy(np_array).cuda()
    else:
        return torch.from_numpy(np_array)


def to_numpy(tensor):
    """ Convert a tensor back to numpy array
    """
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def to_batch_first(tensor):
    """ Convert a tensor from time first to batch first

    Args:
        tensor: [T, B, ...]
    Returns:
        tensor: [B, T, ...]
    """
    return tensor.transpose(0, 1)


def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them
    from their history.

    args:
        h: a Variable
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_state(v) for v in h)


def gather_dim(input_tensor, inds, dim=0):
    """ Gather subset of a tensor on a given dimension with input indices

    Args:
        input_tensor: n-dimensional tensor to gather from
        inds: a numpy array of indices [N]
    Returns:
        gathered dims
    """
    return torch.index_select(input_tensor, dim, to_tensor(inds))


def unsort_dim(seq, sort_inds, dim=0):
    """ Given a sorted sequence tensor, "unsort" the sequence.
    This function is exclusively used in the dynamic_rnn function
    but it must be useful for other functions...right?

    Args:
        seq: sorted sequence (n-dimensional) to unsort
        sort_inds: the order that sequence is sorted
        dim: on which dimension to unsort
    Returns:
        an unsorted sequence of the origina shape
    """
    inv_inds = np.zeros_like(sort_inds)
    for i, ind in enumerate(sort_inds):
        inv_inds[ind] = i
    seq = torch.index_select(seq, dim, to_tensor(inv_inds))
    return seq


def to_one_hot(cls_ind, num_cls):
    """ Convert a integer or a integer array to one-hot vector

    Args:
        cls_inds: class indices, can be either Integer or a Integer array
        num_cls: total number of classes
    Returns:
        vec: a numpy array of shape [?, num_cls]
    """
    if isinstance(cls_ind, int):
        vec = np.zeros(num_cls)
        vec[cls_ind] = 1
    elif isinstance(cls_ind, np.ndarray):
        assert(cls_ind.dtype == np.int64 or cls_ind.dtype == np.int32)
        vec = np.zeros(list(cls_ind.shape) + [num_cls])
        vec_shape = vec.shape
        vec = np.reshape(vec, [-1, num_cls])
        for i, c in enumerate(cls_ind.ravel()):
            vec[i, c] = 1
        vec = np.reshape(vec, vec_shape)
    return vec


def to_batch_seq(x, dtype=np.float32):
    """
    convert a single frame to a batched (size 1)
    sequence and to torch variable
    """
    if isinstance(x, int):
        xdim = []
    else:
        xdim = list(x.shape)
    out = np.zeros([1, 1] + xdim, dtype=dtype)
    out[0, 0, ...] = x
    return to_tensor(out)


def pad_seq_to_max_len(seqs, value=0):
    """
    pad a list of sequence to their max length
    inputs:
        seqs: list([t, ...])
    outputs:
        padded_seq: [B, max(t), ...]
        seq_len: numpy [B]
    """
    batch_size = len(seqs)
    seq_len = np.array([s.size(0) for s in seqs])
    max_seq_len = seq_len.max()

    pad_len = np.zeros((batch_size, 2), dtype=np.int64)
    pad_len[:, 1] = max_seq_len - seq_len

    return pad_seq(seqs, pad_len, value)


def pad_seq(seqs, pad_len, value=0):
    """
    pad a list of sequence to begin and end
    inputs:
        seqs: list([t, ...])
        pad_len: numpy [B, 2]
    outputs:
        padded_seq: [B, T, ...]
        seq_len: numpy [B]
    """
    seq_dim = list(seqs[0].size()[1:])
    seq_len = np.array([s.size(0) for s in seqs])

    padded_seqs = []
    for i, seq in enumerate(seqs):
        bp, ep = pad_len[i]
        padded_seq = []
        if bp == 0 and ep == 0:
            padded_seqs.append(seq.unsqueeze(0))
            continue
        if bp > 0:
            s = [bp] + seq_dim
            pad = to_tensor(np.ones(s, dtype=np.float32) * value)
            padded_seq.append(pad)
        padded_seq.append(seq)
        if ep > 0:
            s = [ep] + seq_dim
            pad = to_tensor(np.ones(s, dtype=np.float32) * value)
            padded_seq.append(pad)

        padded_seq = torch.cat(padded_seq, 0)
        padded_seqs.append(padded_seq.unsqueeze(0))

    padded_seqs = torch.cat(padded_seqs, 0)
    return padded_seqs, seq_len


def truncate_seq(seqs, trunc_inds):
    """
    truncate a sequence tensor to different length
    inputs:
        seqs: [B, T, ...]
        trunc_inds: numpy [B, 2]
    outputs:
        trunc_seqs: list([t, ...])
    """
    trunc_seqs = []
    split_seqs = seqs.split(1, dim=0)
    for i, seq in enumerate(split_seqs):
        b, e = trunc_inds[i]
        tseq = seq.squeeze(0)[b: e, ...]
        trunc_seqs.append(tseq)
    return trunc_seqs
