import numpy as np
import scipy.sparse as sp
import torch


def visit(a, b, row, k, row_in, n):
    """
    a 1阶邻接矩阵
    b 要输出的k阶邻接矩阵，得到是一个上三角矩阵
    row,col : 行和列
    row_in: 最初的行号  [row_in, x]->[x,y] ->[y,z]..  最终得[row_in,z]
            所以要记录最初的行号，即入口
    n: 邻接矩阵的行数或者列数

    """

    for j in range(0, n):
        # 尝试改为 range(1,n)
        m = k
        if a[row, j]:
            m = m - 1
            # print("m={}, 坐标为({},{})".format(m,row,j))
            if (m == 0):
                if (not a[row_in, j]) and (row_in != j):
                    b[row_in, j] = 1
                    # print("{},{}满足要求".format(row_in,j))
                else:
                    m = m + 1
                    continue
            else:
                visit(a, b, j, m, row_in, n)

    return


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj1_dense = adj.todense()
    n = adj1_dense.shape[0]
    a = np.array(adj1_dense, dtype=np.float32)
    adj2_dense = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n):
        # print("从第{}行进入".format(i))
        visit(a, adj2_dense, i, 2, i, n)
    adj2 = sp.coo_matrix(adj2_dense, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj2 = normalize(adj + sp.eye(adj2.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, adj2, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
