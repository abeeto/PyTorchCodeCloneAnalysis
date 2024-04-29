'''
It actually is the CNN + Attention model
We can use this to easily implement CNN, CNN + att + T
'''

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics

from dataset import Dataset
from dataset import collate_fn
import numpy as np

root = '/data/yanjianhao/nlp/torch/torch_NRE/data/'

# maybe a bag at a time
batch_size = 160
trigger = 0



class PCNN_attention(nn.Module):
    def __init__(self, n_rel=53 , vocab_size=114043, word_embed_size=50, pre_word_embeds=None, position_embedding=True):
        super(PCNN_attention, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab_size = vocab_size
        self.n_rel = n_rel
        self.hidden_size = 50
        self.word_embed = nn.Embedding(self.vocab_size, self.word_embed_size)
        self.features_size = word_embed_size

        self.pos_embed_size = 5
        # define the position embedding effective domain
        self.max_len = 60
        # number of output channels for CNN
        self.out_c = 230

        if position_embedding:
            self.features_size += 2 * self.pos_embed_size
            self.pos_embed = nn.Embedding(self.max_len * 2 + 1, self.pos_embed_size)

        # word embedding
        if pre_word_embeds is not None:
            self.pre_word_embed = True
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds), requires_grad=True)

        else:
            self.pre_word_embed = False

        self.atten_sm = nn.Softmax()

        # relation embedding size is the same as CNN's output
        # self.r_embed = nn.Parameter(torch.randn(self.n_rel, self.out_c), requires_grad=True)
        self.r_embed = nn.Embedding(self.n_rel, self.out_c)
        # rel_embed : hidden * n_rel
        self.conv = nn.Conv2d(1, self.out_c, (3, self.features_size))
        # self.atten = nn.Linear(self.out_c, )
        self.linear = nn.Linear(self.out_c, self.n_rel)
        # NLL loss is apllied to logit outputs
        self.pred_sm = nn.LogSoftmax()

        # only use for debugging
        self.grad_list = []
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5, inplace=True)


    def _create_position_embed(self, sent_len, pos1, pos2):
        above = torch.Tensor([self.max_len]).expand(sent_len)
        below = torch.Tensor([-1 * self.max_len]).expand(sent_len)
        # lookup tensor should be all positive
        pf1_lookup = ag.Variable(clip(torch.arange(0, sent_len) - float(pos1), above, below).cuda().long() + self.max_len)
        pf2_lookup = ag.Variable(clip(torch.arange(0, sent_len) - float(pos2), above, below).cuda().long() + self.max_len)
        if trigger:
            print("MAX")
            print(pf1_lookup.max())
            print(pf2_lookup.max())
            print("MIN")
            print(pf1_lookup.min())
            print(pf2_lookup.min())
        pf1 = self.pos_embed(pf1_lookup)
        pf2 = self.pos_embed(pf2_lookup)

        return torch.cat([pf1, pf2], dim=1)

    def forward(self, x, max_sent_len, position_embedding=True, labels=None):

        s = self._create_features_for_bag(x, pe=position_embedding, labels=labels)
        # pdb.set_trace()
        # s.register_hook(self._print_grad)
        pred = self.pred_sm(s)
        return pred

        # features : list, feature : torch tensor


    def _print_grad(self, grad):
        self.grad_list.append(grad)


    def _create_features_for_bag(self, x, pe=True, labels=None):
        # max_len = 0
        # for bag in x:
        #     for item in bag:
        #         if max_len < len(item[2]):
        #             max_len = len(item[2])
        max_len = self.max_len

        # weird..
        # if labels is not None:
        labels_lookup = ag.Variable(labels.cuda())
        # r_embeds = self.r_embed(labels_lookup)

        batch_features = []
        for ix, bag in enumerate(x):
            # for each bag...
            features = []

            for item in bag:
                # pdb.set_trace()
                pos1 = item[0]
                pos2 = item[1]
                sent = item[2]
                sent_len = len(sent) if len(sent) < max_len else max_len

                # Here we need to limit the length of one sentence
                lookup_tensor = ag.Variable(torch.LongTensor(sent[:max_len]).cuda())
                feature = self.word_embed(lookup_tensor)

                # pdb.set_trace()
                if pe:
                    pf = self._create_position_embed(sent_len, pos1, pos2)

                    feature = torch.cat([feature, pf], dim=1)

                # makes features 4D
                # feature = feature.expand(1, 1, feature.size()[0], feature.size()[1])

                feature = feature.unsqueeze(0).unsqueeze(0)
                # pdb.set_trace()
                # first two dims indicate the first
                feature = F.pad(feature, (0, 0, 0, max_len - feature.size()[2]))
                features.append(feature)

            # pdb.set_trace()
            features = torch.cat(features, dim=0)
            # fix dims for features
            # pdb.set_trace()
            # features = self.relu(self.conv(features).squeeze(3))
            features = self.tanh(self.conv(features).squeeze(3))
            # no padding in conv
            features = F.max_pool1d(features, max_len - 2).squeeze(2)

            # alternative way
            # only use prediction label.
            # maybe there's intersection for relations so that each relation got a probability

            # bag_size * n_rel
            # for each relation
            # pdb.set_trace()
            atten_weights = self.atten_sm(torch.matmul(self.r_embed.weight, features.t()))
            # pdb.set_trace()
            features = torch.matmul(atten_weights, features)
            if self.dropout is not None:
                features = self.dropout(features)
            # features = self.linear(features)
            features = torch.matmul(features, self.r_embed.weight.t()).diag()
            # if features.size()[0] != 1:
            #     print(ix)
            # pdb.set_trace()
            batch_features.append(features)

        # pdb.set_trace()
        return torch.stack(batch_features)


def one_hot(ids, n_rel):
    """
    ids: numpy array or list shape:[batch_size,]
    n_rel: # of relation to classify
    """
    labels = np.zeros((ids.shape[0], n_rel))
    labels[np.arange(ids.shape[0]), ids] = 1
    return labels


def clip(x, above, below):
    # use two mask to dsefine bounds
    # pdb.set_trace()
    mask_above = (above >= x).float()
    tmp = mask_above * x + (1 - mask_above) * above

    mask_below = (below <= tmp).float()
    tmp = mask_below * tmp + (1 - mask_below) * below
    return tmp


def train():
    datasets = Dataset(root)
    loader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    word_embed = datasets.vecs
    model = PCNN_attention(pre_word_embeds=word_embed)
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # Turn off the updates for word_embeddings
    # optimizer = optim.SGD(iter([module[1] for module in model.named_parameters()
    #                             if module[0] != "word_embed.weight"]), lr=1e-2)
    n_epoches = 25

    for i in range(n_epoches):
        print("For epoch %d:"%i)
        acm_loss = 0
        for batch_ix, (bags, labels) in enumerate(loader):
            # print('batch_ix is : %d'%batch_ix)
            model.zero_grad()
            # if batch_ix < 431:
            #     continue
            # if batch_ix == 431:
            #     pdb.set_trace()
            out = model(bags, datasets.max_sent_len, labels=labels)
            # pdb.set_trace()

            # compute negative likelihood loss
            labels = ag.Variable(labels.cuda())
            loss = loss_func(out, labels)
            # pdb.set_trace()
            loss.backward()

            optimizer.step()
            acm_loss += loss
            if (batch_ix + 1) % 50 == 0:
                ave_loss = acm_loss / 50
                print("For batch id %d:" %batch_ix)
                print("The average loss is %s"%(ave_loss))
                acm_loss = 0

    # saves the results
    torch.save(model.state_dict(), "/data/yanjianhao/nlp/torch/torch_NRE/model/att+cnn.dat")


def test(PATH):
    test_data = Dataset(root, train_test='test')
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    word_embed = test_data.vecs
    model = PCNN_attention(pre_word_embeds=word_embed)
    model.load_state_dict(torch.load(PATH))
    model.cuda()
    # in test, the r embed need to amplify with the p of dropout
    model.r_embed.weight = nn.Parameter(model.r_embed.weight.data / model.dropout.p)
    model.eval()

    prec = []
    outs = []
    y_true = []

    for batch_ix, (bags, labels) in enumerate(test_loader):
        # if batch_ix > 10:
        #     break
        out = model(bags, test_data.max_sent_len, labels=labels)
        # pdb.set_trace()
        outs.append(out.cpu().data.numpy())
        y_true.append(labels.numpy())
        pred = out.max(dim=1)[1].long().data
        bz = pred.size()[0]
        correct = pred.eq(labels.cuda())
        acc = float(correct.sum()) / bz
        prec.append(acc)
        print("Accuracy in %d batch:%f"%(batch_ix, acc))

    prec = sum(prec) / len(prec)
    print("Average test accuracy is %f"%prec)

    # draw precision-recall curve

    y_pred = np.concatenate(outs, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    np.save('./result/labels_word_embed_keep.npy', y_true)
    np.save("./result/prediction_word_embed_keep.npy", y_pred)


    # pdb.set_trace()


if __name__ == "__main__":
    train()
    path = "/data/yanjianhao/nlp/torch/torch_NRE/model/att+cnn.dat"
    test(path)









