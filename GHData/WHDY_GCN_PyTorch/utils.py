import numpy as np


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    D = np.power(rowsum, -0.5).flatten()
    D[np.isinf(D)] = 0.0
    DMat = np.diag(D)
    return adj.dot(DMat).transpose().dot(DMat)


def prerocess_adj(adj):
    adjAddSelfLoop = adj + np.eye(adj.shape[0])
    adjNormalized = normalize_adj(adjAddSelfLoop)
    return adjNormalized
