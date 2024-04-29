import math
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import scipy.sparse as sp

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
    # 单条数据格式： paper id + features + label
    #   第0维是paper id
    #   最后一维是label
    #   1:-1范围是feature，标识关键词出现与否
    id_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    # 用scipy的稀疏矩阵储存features
    features = sp.csr_matrix(id_features_labels[:, 1:-1], dtype=np.float32)
    # 对label做one-hot编码
    labels = encode_onehot(id_features_labels[:, -1])

    # build graph
    # paper id的array
    idx = np.array(id_features_labels[:, 0], dtype=np.int32)
    # paper_id:index_in_array的map
    id_map = {j: i for i, j in enumerate(idx)} 
    # 边表示论文之间的引用关系，<ID of cited paper> <ID of citing paper>，有向图
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
    '''
    map(function, iterable, ...)会根据提供的函数function对指定序列iterable做映射
        例如map(math.sin, [1,2,3,4,5]) -> 计算列表各个元素的sin值
    '''
    # 利用id_map，将paper_id-paper_id关系转换为index_in_array-index_in_array关系
    edges = np.array(list(map(id_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    # 根据边的连接关系构建邻接矩阵，有向图，非对称
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    
    # print(np.min(adj-adj.T),np.max(adj-adj.T)) # 是否对称
    # tmp1 = sparse_mx_to_torch_sparse_tensor(adj).to_dense().numpy() # scipy稀疏矩阵 -> torch稀疏矩阵 -> torch dense矩阵 -> numpy
    # tmp2 = tmp1 + np.multiply(tmp1.T,tmp1.T>tmp1) - np.multiply(tmp1,tmp1.T>tmp1);print(np.max(tmp2-tmp2.T),np.min(tmp2-tmp2.T))
    
    '''
    这里有一个问题，为什么不用论文中说的对称归一化的拉普拉斯矩阵？而是先对称化，再归一化？
    '''

    # 不同于dot()点乘，即常见的矩阵相乘，multiply() 是矩阵对应元素相乘，即Hadamard乘积
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 对称化，将非对称位置改为对称，0->1

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # 自连接+归一化


    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """按行归一化，元素/该元素所在行元素的和"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''numpy版本
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype(np.float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """scipy稀疏矩阵 -> torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training) # 对x进行dropout
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1) # 最后一层用log_softmax激活函数需要和nll_loss搭配，而CrossEntropyLoss = softmax+log+NLLLoss 一步到位
        return x


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.2)
optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)

'''
model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
'''

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(200):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

'''
[SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
'''