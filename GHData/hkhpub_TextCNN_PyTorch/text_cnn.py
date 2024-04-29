# _*_ coding: UTF8 _*_
"""
code reference: https://github.com/junwang4/CNN-sentence-classification-pytorch-2017/blob/master/cnn_pytorch.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.autograd as autograd
import data_helpers
from sklearn.model_selection import KFold
import time

use_cuda = torch.cuda.is_available()
embedding_dim = 300
num_filters = 100
filter_sizes = [3, 4, 5]
batch_size = 50
num_epochs = 10

np.random.seed(0)
torch.manual_seed(0)

X, Y, word_to_ix, ix_to_word = data_helpers.load_data()

vocab_size = len(word_to_ix)
max_sent_len = X.shape[1]
num_classes = Y.shape[1]

print('vocab size       = {}'.format(vocab_size))
print('max sentence len = {}'.format(max_sent_len))
print('num of classes   = {}'.format(num_classes))


class TextCNN(nn.Module):

    def __init__(self, max_sent_len, embedding_dim, filter_sizes, num_filters, vocab_size, num_classes):
        '''
        :param embedding_dim:
        :param filer_sizes: list 형식 -> e.g. [3, 4, 5]
        :param num_filters: size별 filter의 갯수
        :param vocab_size:
        :param target_size:
        '''
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # do we really need this?
        # (명시적으로 non-static word embedding 임을 나타내기 위함)
        self.word_embeddings.weight.requires_grad = True

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_kernel_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=filter_size)
            # TODO: Sequential 스터디
            component = nn.Sequential(
                conv1,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size))

            if use_cuda:
                component = component.cuda()
            conv_blocks.append(component)

        # TODO: ModuleList 스터디
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, sentence_len)
        x = self.word_embeddings(x)
        # 기존 x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)   # switch 2nd and 3rd axis
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        # x_list.shape: [(num_filters, filter_size_3), (num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)      # concatenate along filter_sizes
        out = out.view(out.size(0), -1)
        # feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1)


def evaluate(model, x_test, y_test):
    inputs = autograd.Variable(x_test)
    preds = model(inputs)
    preds = torch.max(preds, 1)[1]
    y_test = torch.max(y_test, 1)[1]
    if use_cuda:
        preds = preds.cuda()
    eval_acc = sum(preds.data == y_test) * 1.0 / len(y_test)
    return eval_acc


def train_test_one_split(train_index, test_index):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    # numpy array to torch tensor
    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).float()
    dataset_train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).float()
    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    model = TextCNN(max_sent_len=max_sent_len,
                    embedding_dim=embedding_dim,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    vocab_size=vocab_size,
                    num_classes=num_classes)

    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()       # set the model to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            preds = model(inputs)
            if use_cuda:
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()        # set the model to evaluation mode
        eval_acc = evaluate(model, x_test, y_test)
        print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}'.format(epoch, loss.data[0], eval_acc))

    model.eval()        # set the model to evaluation mode
    eval_acc = evaluate(model, x_test, y_test)
    return eval_acc


cv_folds = 5    # 5-fold cross validation
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
acc_list = []
tic = time.time()
for cv, (train_index, test_index) in enumerate(kf.split(X)):
    acc = train_test_one_split(train_index, test_index)
    print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
    acc_list.append(acc)
print('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format(sum(acc_list)/len(acc_list), time.time()-tic))