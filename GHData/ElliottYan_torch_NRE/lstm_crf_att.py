import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics

from dataset import Dataset, Temporal_Data
from dataset import collate_fn
import numpy as np

torch.manual_seed(1)


root = '/data/yanjianhao/nlp/torch/torch_NRE/data/'


class Lstm_crf_att(nn.Module):
    def __init__(self, n_rel=53 , vocab_size=114043, word_embed_size=50, pre_word_embeds=None, position_embedding=True):
        super(Lstm_crf_att, self).__init__()
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
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


    def forward(self, x, labels):
        # batch of sentences
        # list of cnn outputs
        s = self._create_features_for_bag(x, labels=labels)

        pdb.set_trace()

        # nn.utils.rnn.pad_packed_sequence(s, batch_first=True)
        return


    def _create_features_for_bag(self, x, pe=True, labels=None):
        # max_len = 0
        # for bag in x:
        #     for item in bag:
        #         if max_len < len(item[2]):
        #             max_len = len(item[2])
        max_len = self.max_len

        # this is for rnn padding
        max_bag_len = 0
        for bag in x:
            if max_bag_len < len(bag):
                max_bag_len = len(bag)

        # weird..
        if labels is not None:
            labels_lookup = ag.Variable(labels.cuda())
            r_embeds = self.r_embed(labels_lookup)

        batch_features = []
        for ix, bag in enumerate(x):
            # for each bag...
            features = []

            for item in bag:
                # Now each item in bag is Mention objects

                pos1 = item.pos[0]
                pos2 = item.pos[1]
                sent = item.sent
                sent_len = len(sent) if len(sent) < max_len else max_len
                time = item.time

                try:
                    # Here we need to limit the length of one sentence
                    lookup_tensor = ag.Variable(torch.LongTensor(sent[:max_len]).cuda())
                    feature = self.word_embed(lookup_tensor)
                except:
                    pdb.set_trace()
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

            # features -> word_embed + ps_embed : for each bag
            # each feature in features are (1, out_c) size
            features = torch.cat(features, dim=0)
            # fix dims for features
            features = self.relu(self.conv(features).squeeze(3))
            # no padding in conv
            pdb.set_trace()
            features = F.max_pool1d(features, max_len - 2).squeeze(2)
            batch_features.append(F.pad(features, (0, 0, 0, max_bag_len - features.size()[2])))


        return batch_features


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


def train():
    datasets = Temporal_Data(root)
    loader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    word_embed = datasets.vecs
    model = Lstm_crf_att(pre_word_embeds=word_embed)
    if torch.cuda.is_available():
        model = model.cuda()


    for batch_ix ,(bags, label) in loader:
        out = model.forward(bags, labels)







def main():
    train()


if __name__ == "__main__":
    main()
