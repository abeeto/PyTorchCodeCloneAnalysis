import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from models.CNNEncoder import CNNEncoder
from utils import print, ProgressBar

class SentenceClassifier(nn.Module):

    def __init__(self, config, word_embedding_array, dictionary):
        super(SentenceClassifier, self).__init__()
        self.config = config
        self.build_model(word_embedding_array, dictionary)

    def build_model(self, word_embedding_array, dictionary):
        self.encoder = CNNEncoder(self.config, word_embedding_array, dictionary)
        self.pred = nn.Linear(len(self.config.cnn_n_gram_list)*self.config.cnn_output_channel, self.config.classifier_class_number)
        self.drop = nn.Dropout(self.config.classifier_dropout_rate)
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range=0.1):
        #self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)
        self.pred.weight = xavier_normal(self.pred.weight)

    def forward(self, inp):
        outp = self.encoder.forward(inp) # [batch_size, len(self.config.cnn_n_gram_list)*self.config.cnn_output_channel]
        pred = self.softmax(self.pred(self.drop(outp)))
        return pred

    def encode(self, inp):
        return self.encoder.forward(inp)

def run_SentenceClassifier(config, model, train_data, train_label, test_data, test_label):

    """Closure"""

    def train(data, label, epoch):
        assert data.shape[0] == label.shape[0]
        datasize = label.shape[0]
        indexes = np.random.permutation(datasize)
        updatetime = int(datasize//config.classifier_batch_size)+1
        total_loss = 0
        if config.show_progress:
            bar = ProgressBar('Classifier Train epoch {} / {}'.format(epoch,config.classifier_epochs), max=updatetime)
        for i in range(updatetime):
            pos = i*config.classifier_batch_size
            ids = indexes[pos:(pos+config.classifier_batch_size) if (pos+config.classifier_batch_size) < datasize else datasize]
            current_batch_size = len(ids)
            batch_data = Variable(torch.from_numpy(data[ids]))
            batch_label = Variable(torch.from_numpy(label[ids]))
            if config.cuda:
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
            output = model(batch_data)
            assert output.size(0) == current_batch_size
            loss = F.cross_entropy(output, batch_label)
            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), config.classifier_grad_norm_clip)
            optimizer.step()
            if config.show_progress:
                bar.next()
        if config.show_progress:
            bar.finish()
        return total_loss[0]

    def test(data, label):
        assert data.shape[0] == label.shape[0]
        datasize = label.shape[0]
        updatetime = int(datasize//config.classifier_batch_size)+1
        total_loss = 0
        total_acc = 0
        indexes = np.arange(datasize)
        for i in range(updatetime):
            pos = i*config.classifier_batch_size
            ids = indexes[pos:(pos+config.classifier_batch_size) if (pos+config.classifier_batch_size) < datasize else datasize]
            current_batch_size = len(ids)
            batch_data = Variable(torch.from_numpy(data[ids]))
            batch_label = Variable(torch.from_numpy(label[ids]))
            if config.cuda:
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
            output = model(batch_data)
            loss = F.cross_entropy(output, batch_label)
            total_loss += loss.data
            predictions = torch.max(output,1)[1].type_as(batch_label) # axis1での，つまり各データの中で最大の要素のindexを返す．(argmax)
            total_acc += predictions.eq(batch_label).cpu().sum().data[0]
        return total_loss[0], total_acc/datasize

    if config.show_progress:
        print("Model parameters")
        for p in model.parameters():
            print("{}, {}".format(p.size(), p.requires_grad))

    # filter that only require gradient decent
    parameters = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        parameters.append(p)

    optimizer = optim.Adam(parameters, lr=config.classifier_lr)

    for epoch in range(config.classifier_epochs):
        model.train()
        train_total_loss = train(train_data, train_label, epoch+1)
        model.eval()
        test_total_loss, test_acc = test(test_data, test_label)
        print("Train Epoch: {:3d} \t train_total_loss: {:.4f} \t test_total_loss: {:.4f} \t test_acc: {:.3f}".format(epoch + 1, train_total_loss, test_total_loss, test_acc))
