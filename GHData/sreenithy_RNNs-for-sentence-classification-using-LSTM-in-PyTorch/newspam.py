import torch
from dataset import load_dataset
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
torch.manual_seed(1)

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.embedding_dim=embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        if self.use_gpu:
            embeds = self.word_embeddings(sentence.cuda())
        else:
            embeds=self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size,-1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])

        return y


def train(model,epochs):
    print('starting training')
    Al=0.0
    Aa=0.0
    train_iter, test_iter = load_dataset('spam', batchsize)
    #model.train(True)
    for epo in range(epochs):
        avg_loss = 0.0
        avg_acc = 0.0
        print('Epoch {}/{}'.format(epo, epochs - 1))
        print('-' * 10)
        i = 0
        truth_res = []
        pred_res = []
        #model.
        for batch in train_iter:
            sentence = batch.sentence
            label = batch.label
            print ("Batch :%d" % (i))

            i = i + 1
            truth_res += list(label.data)

            model.zero_grad()
            model.batch_size = np.shape(sentence)[1]
            model.hidden = model.init_hidden()
            tag_scores = model(sentence)
            if USE_GPU:
                 loss = loss_function(tag_scores, label.cuda())
            else:
                 loss=loss_function(tag_scores,label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            pred_label = tag_scores.cpu().data.max(1)[1].numpy()

            pred_res += [x for x in pred_label]
            acc = get_accuracy(truth_res, pred_res)
            avg_acc+=acc

        print('The average loss after completion of %d epochs is %g' % ((epo + 1), (avg_loss / i)))
        Al+= (avg_loss/i)
        print('The average accuracy after completion of %d epochs is %g' % ((epo + 1), (avg_acc / i)))
        Aa += (avg_acc / i)
    s=Al/epochs
    t=Aa/epochs

    print('Training  Loss:')
    print(s)
    print("Training  Accuracy:")
    print(t)
    torch.save(model.state_dict(), 'spam.pth')
    return model

def test(model):
    train_iter, test_iter = load_dataset('spam', batchsize)
    print('Starting testing')
    truth_res = []
    pred_res = []
    avg_loss = 0.0
    avg_acc = 0.0
    i=0
    for batch in test_iter:
        sentence = batch.sentence
        label = batch.label
        print ("Batch :%d" % (i))

        i = i + 1
        truth_res += list(label.data)

        model.zero_grad()
        model.batch_size = np.shape(sentence)[1]
        model.hidden = model.init_hidden()
        tag_scores = model(sentence)
        pred_label = tag_scores.cpu().data.max(1)[1].numpy()

        pred_res += [x for x in pred_label]
        acc = get_accuracy(truth_res, pred_res)
        avg_acc += acc
    t=avg_acc/i
    print('Test Classification accuracy')
    print(t)



vocabsize=28049
embeddingsize=500
batchsize=128
hiddendimsize=50
labelsize=2

USE_GPU = torch.cuda.is_available()
model = LSTMClassifier(embeddingsize,hiddendimsize,vocabsize,labelsize, batchsize,USE_GPU)
if USE_GPU:
    model = model.cuda()
#model=model.cuda()
print(model)

loss_function = nn.CrossEntropyLoss()
if USE_GPU:
     loss_function=loss_function.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3,eps=0.1)
model=train(model,epochs=50)
test(model)

