from os import path
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from utils import make_batch, get_batch, plot_loss_fig


class MyModel(nn.Module):
    def __init__(self, opt, dict_len):
        super(MyModel, self).__init__()
        self.embed = nn.Embedding(dict_len, opt.embed_dim)
        self.drop = nn.Dropout(opt.drop)
        self.opt = opt
        if opt.rnn == 'LSTM':
            self.encoder = nn.LSTM(input_size=opt.embed_dim, hidden_size=opt.hid_dim, num_layers=opt.layer_num,
                                   batch_first=True, dropout=opt.rnn_drop, bidirectional=opt.rnn_bidir)
        else:
            assert opt.rnn == 'GRU'
            self.encoder = nn.GRU(input_size=opt.embed_dim, hidden_size=opt.hid_dim, num_layers=opt.layer_num,
                                   batch_first=True, dropout=opt.rnn_drop, bidirectional=opt.rnn_bidir)
        if opt.rnn_bidir is True:
            self.decoder = nn.Linear(opt.hid_dim * 2, dict_len)
        else:
            self.decoder = nn.Linear(opt.hid_dim, dict_len)

        if opt.tie_weights:
            if opt.embed_dim != opt.hid_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.embed.weight

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        input_emb = self.drop(self.embed(input))
        output, hidden = self.encoder(input_emb, hidden)
        decoded = self.decoder(output.contiguous().view(-1, output.size(2)))
        return decoded, hidden

    def init_hidden(self, batch_size):
        bidirectional = 2 if self.opt.rnn_bidir else 1
        h = torch.zeros(bidirectional * self.opt.layer_num, batch_size, self.opt.hid_dim)
        if self.opt.rnn == 'GRU':
            return Variable(h.cuda()) if self.opt.use_cuda else Variable(h)
        else:
            c = torch.zeros(bidirectional * self.opt.layer_num, batch_size, self.opt.hid_dim)
            return (Variable(h.cuda()), Variable(c.cuda())) if self.opt.use_cuda else (Variable(h), Variable(c))


def np2tensor(opt, data):
    return Variable(torch.from_numpy(data).cuda()) if opt.use_cuda else Variable(torch.from_numpy(data))


def train(opt, corpus):
    train_data = make_batch(corpus.train_data, opt.batch_size)
    valid_data = make_batch(corpus.val_data, opt.batch_size)
    model = MyModel(opt, len(corpus.dictionary))
    if opt.use_cuda:
        model.cuda()
    # Here we use summation instead of avg for the loss of each mini-batch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=opt.init_lr)

    print_loss, plot_loss, plot_losses, plot_eval = 0.0, 0.0, [], []
    for epoch in range(1, opt.epoch_num + 1):
        model.train()
        hidden = model.init_hidden(opt.batch_size)
        for i in range(0, train_data.shape[1] - 1, opt.bptt_len):
            origin, target = get_batch(opt, train_data, i)
            origin = np2tensor(opt, origin)
            target = np2tensor(opt, target)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)

            optimizer.zero_grad()
            predict, hidden = model(origin, hidden)
            loss = criterion(predict, target)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            print_loss += loss.data[0]
            plot_loss += loss.data[0]

        if epoch % opt.print_span == 0:
            print_loss_avg = print_loss / opt.print_span / train_data.shape[1]
            print_loss = 0.0
            print("Epoch {0} (total {1} epochs) | AvgLoss {2}".format(epoch, opt.epoch_num, print_loss_avg))

        if epoch % opt.plot_span == 0:
            plot_loss_avg = plot_loss / opt.plot_span / train_data.shape[1]
            plot_losses.append(plot_loss_avg)
            plot_loss = 0.0

        if epoch % opt.eval_span == 0:
            eval_loss = evaluate(opt, valid_data, model, criterion)
            plot_eval.append(eval_loss)
            print('-' * 89)
            print("Validation: Epoch {0} (total {1} epochs) | EvaluateLoss {2}".format(epoch, opt.epoch_num, eval_loss))

        if epoch % opt.checkpoint == 0:
            torch.save(model, path.join(opt.save_dir, 'model.cpt'))
            print("The model has been saved in {0}".format(opt.save_dir))
            with open(path.join(opt.save_dir, 'train_loss'), 'wb') as handle:
                pickle.dump(plot_losses, handle)
            with open(path.join(opt.save_dir, 'eval_loss'), 'wb') as handle:
                pickle.dump(plot_eval, handle)
            print("The list for plotting training loss and eval loss saved in {0}".format(opt.save_dir))

    plot_loss_fig(opt, plot_losses, plot_eval)

    # Test the performance on testing dataset
    test_data = make_batch(corpus.test_data, opt.batch_size)
    test_loss = evaluate(opt, test_data, model, criterion)
    print('-' * 89)
    print("Test: TestLoss {0}".format(test_loss))

    torch.save(model, path.join(opt.save_dir, 'final_model.cpt'))
    print("Training finished, the fianl model has been saved in {0}".format(opt.save_dir))


def evaluate(opt, valid_data, model, criterion):
    accu_loss = 0.0
    model.eval()
    hidden = model.init_hidden(opt.batch_size)
    for i in range(0, valid_data.shape[1] - 1, opt.bptt_len):
        origin, target = get_batch(opt, valid_data, i)
        origin = np2tensor(opt, origin)
        target = np2tensor(opt, target)
        hidden = repackage_hidden(hidden)

        predict, hidden = model(origin, hidden)
        loss = criterion(predict, target)
        accu_loss += loss.data[0]

    accu_loss /= valid_data.shape[1]
    return accu_loss


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def generate_text(opt, corpus):
    assert opt.temperature >= 1e-3

    model = torch.load(path.join(opt.save_dir, 'final_model.cpt'))
    model.eval()
    if opt.use_cuda:
        model.cuda()
    else:
        model.cpu()

    dict_len = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    origin = torch.rand(1, 1).mul(dict_len).long()
    origin = Variable(origin.cuda()) if opt.use_cuda else Variable(origin)

    with open(path.join(opt.save_dir, 'generate.txt'), 'w') as f:
        for i in range(opt.generate_word_num):
            predict, hidden = model(origin, hidden)
            word_weights = predict.squeeze().data.div(opt.temperature).exp()
            word_idx = torch.multinomial(word_weights, 1)[0]
            origin.data.fill_(word_idx)
            word = corpus.dictionary.idx2w[word_idx]

            f.write('\n\n' if word == '<eos>' else word + ' ')

