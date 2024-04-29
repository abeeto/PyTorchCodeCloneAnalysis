import torch
import torch.nn as nn
import func
import data
import rnn
import utils
import os
import layers
from contextualized_embedding import ElmoEmbedding

class Model(nn.Module):
    def without_embedding(self, word_vocab_size, word_dim, char_vocab_size, char_dim):
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=data.NULL_ID)
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=data.NULL_ID)
        self.word_dim = word_dim


    def with_embedding(self, word_mat, char_mat):
        self.word_embedding = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.word_dim = self.word_embedding.weight.shape[1]
        self.char_embedding = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.char_embedding.padding_idx = data.NULL_ID


    def with_contextualized_embedding(self, embedding_dim, word_dim, char_vocab_size, char_dim):
        self.elmo = ElmoEmbedding(embedding_dim)
        if embedding_dim != word_dim:
            self.dense_word = nn.Sequential()
            dim = embedding_dim
            while True:
                prev_dim = dim
                dim = dim // 2
                if dim < word_dim:
                    break
                linear = nn.Linear(prev_dim, dim)
                relu = nn.LeakyReLU(0.1)
                norm = nn.BatchNorm1d(dim)
                dropout = nn.Dropout(0.2)
                self.dense_word.add_module(f'linear{dim}', linear)
                self.dense_word.add_module(f'norm{dim}', norm)
                self.dense_word.add_module(f'relu{dim}', relu)
                self.dense_word.add_module(f'dropout{dim}', dropout)
            self.dense_word.add_module(f'linear{word_dim}', nn.Linear(prev_dim, word_dim))
            self.dense_word.add_module(f'norm{word_dim}', nn.BatchNorm1d(word_dim))
        else:
            self.dense_word = lambda x:x
        self.word_dim = word_dim
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=data.NULL_ID)


    def initialize(self, char_hidden_size, encoder_hidden_size, rnn_type, dropout):
        '''
        char_hidden_size: default 200
        '''
        self.dropout = nn.Dropout(dropout)
        self.encoder_hidden_size = encoder_hidden_size
        self.rnn_type = nn.GRU if rnn_type == 'gru' else nn.LSTM
        encoder_layers = 3
        #char encoding
        if char_hidden_size != 0:
            self.char_rnn = rnn.RNNEncoder(
                input_size=self.char_embedding.weight.shape[1],
                num_layers=1,
                hidden_size=char_hidden_size,
                bidirectional=True,
                type='gru')
        '''
        self.char_rnn = rnn.StackedBRNN(
            input_size=self.word_dim,
            hidden_size=char_hidden_size,
            num_layers=1,
            rnn_type=nn.GRU,
            concat_layers=True,
            padding=True,
            dropout_rate=dropout)
        '''
        #encoding
        self.encoder = rnn.StackedBRNN(
            input_size=self.word_dim+char_hidden_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            rnn_type=self.rnn_type,
            concat_layers=True,
            padding=True,
            dropout_rate=dropout)
        self.encoder_size = encoder_layers*2*encoder_hidden_size
        #attention
        self.pq_attention = layers.DotAttention(
            input_size=self.encoder_size,
            memory_size=self.encoder_size,
            hidden_size=encoder_hidden_size,
            dropout=dropout)
        self.pq_encoder = rnn.StackedBRNN(
            input_size=self.encoder_size*2,
            hidden_size=encoder_hidden_size,
            num_layers=1,
            rnn_type=self.rnn_type,
            concat_layers=True,
            padding=True,
            dropout_rate=dropout)
        self.pq_attention_size = encoder_hidden_size*2
        #match
        self.match_attention = layers.DotAttention(
            input_size=self.pq_attention_size,
            memory_size=self.pq_attention_size,
            hidden_size=encoder_hidden_size,
            dropout=dropout)
        self.match_encoder = rnn.StackedBRNN(
            input_size=self.pq_attention_size*2,
            hidden_size=encoder_hidden_size,
            num_layers=1,
            rnn_type=self.rnn_type,
            concat_layers=True,
            padding=True,
            dropout_rate=dropout)
        self.match_size = encoder_hidden_size*2
        #pointer
        self.summary = layers.Summary(
            memory_size=self.match_size,
            hidden_size=encoder_hidden_size,
            dropout=dropout)
        self.pointer_net = layers.PointerNet(
            match_size=self.match_size,
            hidden_size=encoder_hidden_size,
            dropout=dropout)


    def forward(self, c, q, ch, qh, ct=None, qt=None):
        n, pl, _ = ch.shape

        #char encoding
        ql = qh.shape[1]
        c_emb = self.convert_word_embedding(ct, c)
        q_emb = self.convert_word_embedding(qt, q)

        if hasattr(self, 'char_rnn'):
            ch = ch.view(n*pl, -1)
            qh = qh.view(n*ql, -1)
            ch_len = (ch != data.NULL_ID).sum(-1)
            qh_len = (qh != data.NULL_ID).sum(-1)

            ch_emb = self.char_embedding(ch)#[n*pl, cl, dc]
            qh_emb = self.char_embedding(qh)#[n*ql, cl, dc]
            ch_emb = self.dropout(ch_emb)
            qh_emb = self.dropout(qh_emb)

            _, state = self.char_rnn(ch_emb, ch_len)#[num_layers,n*pl,dc]
            ch_emb = torch.cat([state[0], state[1]], -1).view(n, pl, -1)
            _, state = self.char_rnn(qh_emb, qh_len)
            qh_emb = torch.cat([state[0], state[1]], -1).view(n, ql, -1)

            #encoding
            c_emb = torch.cat([c_emb, ch_emb], -1)#[n, pl, dw+dc=500]
            q_emb = torch.cat([q_emb, qh_emb], -1)#[n, ql, dw+dc=500]

        c_len = (c != data.NULL_ID).sum(-1)
        q_len = (q != data.NULL_ID).sum(-1)
        c_mask = func.sequence_mask(c_len)
        q_mask = func.sequence_mask(q_len)
        c = self.encoder(c_emb, c_mask)
        q = self.encoder(q_emb, q_mask)
        #attention
        qc_att = self.pq_attention(c, q, q_mask)
        att = self.pq_encoder(qc_att, c_mask)
        #match
        self_att = self.match_attention(att, att, c_mask)
        match = self.match_encoder(self_att, c_mask)
        #pointer
        init = self.summary(q[:,:,-2*self.encoder_hidden_size:], q_mask)
        logits1, logits2 = self.pointer_net(init, match, c_mask)
        return logits1, logits2


    def convert_word_embedding(self, text, ids):
        if hasattr(self, 'elmo'):
            emb = self.elmo.convert(text)
            batch, length, dim = emb.shape
            return self.dense_word(emb.view(batch*length, dim)).view(batch, length, -1)
        else:
            return self.word_embedding(ids)


    def calc_span(self, logits1, logits2):
        outer = torch.bmm(nn.functional.softmax(logits1, dim=-1).unsqueeze(2), nn.functional.softmax(logits2, dim=-1).unsqueeze(1))
        ms = [m.tril(15).triu() for m in torch.unbind(outer, 0)]
        outer = torch.stack(ms, 0)
        yp1 = outer.max(2)[0].max(1)[1]
        yp2 = outer.max(1)[0].max(1)[1]
        return yp1, yp2


class PositionLoss(nn.Module):
    def __init__(self):
        super(PositionLoss, self).__init__()
        self.lsm = nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss(size_average=True)


    def forward(self, logits, target):
        output = self.lsm(logits)
        loss = self.criterion(output, target)
        return loss


def build_train_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = build_model(opt, dataset)
    feeder = data.TrainFeeder(dataset, opt.batch_size, opt.char_limit)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
    feeder.prepare('train')
    return model, optimizer, feeder


def build_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = Model()
    if opt.dataset == 'squad':
        if opt.with_elmo == 1:
            model.with_contextualized_embedding(opt.embedding_dim, opt.word_dim, len(dataset.char_emb), opt.char_dim)
        else:
            model.with_embedding(func.tensor(dataset.word_emb), func.tensor(dataset.char_emb))
    else:
        model.without_embedding(len(dataset.word_emb), opt.word_dim, len(dataset.char_emb), opt.char_dim)
    model.initialize(opt.char_hidden_size, opt.encoder_hidden_size, opt.rnn_type, opt.dropout)
    if func.gpu_available():
        model = model.cuda()
    return model


def make_loss_compute():
    criterion = PositionLoss()
    if func.gpu_available():
        criterion = criterion.cuda()
    return criterion


def load_or_create_models(opt, train):
    if os.path.isfile(opt.ckpt_path):
        ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
        model_options = ckpt['model_options']
        for k, v in model_options.items():
            setattr(opt, k, v)
            print('-{}: {}'.format(k, v))
    else:
        ckpt = None
    if train:
        model, optimizer, feeder = build_train_model(opt)
    else:
        model = build_model(opt)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        if train:
            optimizer.load_state_dict(ckpt['optimizer'])
            feeder.load_state(ckpt['feeder'])
    if train:
        return model, optimizer, feeder, ckpt
    else:
        return model, ckpt


def restore(opt, model, optimizer, feeder):
    ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if feeder is not None:
        feeder.load_state(ckpt['feeder'])


def save_models(opt, model, optimizer, feeder):
    model_options = ['char_hidden_size', 'encoder_hidden_size', 'rnn_type']
    model_options = {k:getattr(opt, k) for k in model_options}
    utils.ensure_folder(opt.ckpt_path)
    torch.save({
        'model':  model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'feeder': feeder.state(),
        'model_options': model_options
        }, opt.ckpt_path)


if __name__ == '__main__':
    model = Model()
    model.init_with_embedding(func.tensor([[1, 2, 3], [4, 5, 6]]).float(), func.tensor([[1, 2], [3, 4]]).float())
    assert not model.word_embedding.weight.requires_grad
    assert model.char_embedding.weight.requires_grad