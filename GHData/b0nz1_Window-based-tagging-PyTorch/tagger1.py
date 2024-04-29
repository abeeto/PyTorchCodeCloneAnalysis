import sys
import numpy
import os
import torch
from random import shuffle
import matplotlib.pyplot as pyplot
import functools
"""
Igal Zaidman 311758866
Yehuda Gihasi 305420671
"""

class mainTagger:
    #Initialization function - creates ngrams and vocabulary
    def __init__(self, file_name, with_features,vectors_file = "", vocab_file = ""):
        self.w2i = {}
        self.t2i = {}
        self.i2t = {}
        self.i2w = {}
        self.with_features = with_features
        self.tags = None
        self.vectors = None
        self.middle = int(NGRAM / 2)
        self.ngrams = []
        
        file = open(file_name, "r")
        content = file.read().split('\n')
        
        #<s> symbolizes the start of a line and </s> the end  
        sentences = [("<s>", "s"), ("<s>", "s")]
        for row in content:
            if row == "":
                sentences.append(("</s>", "s"))
                sentences.append(("</s>", "s"))
                sentences.append(("<s>", "s"))
                sentences.append(("<s>", "s"))
            else:
                sp = row.split()
                sentences.append((sp[0], sp[1] if len(sp) > 1 else ''))
        sentences.append(("<s>", "s"))
        sentences.append(("<s>", "s"))
        self.sentences = sentences
        
        #create all the ngrams from the text. Treat each line seperately.
        for i in range(len(sentences) - NGRAM):
            if sentences[i+self.middle][1] != 's':
                self.ngrams.append((list(map(lambda w: w[0], self.sentences[i:i + NGRAM])), self.sentences[i + self.middle][1]))        
        
        if vectors_file:
            self.vectors = list(numpy.loadtxt(vectors_file))
            vocabWords = open("vocab.txt", "r").read().lower().split('\n')
            self.vocab = [word for word in vocabWords if word != '']    
        else:    
            self.vocab = list(set([w[0].lower() for w in self.sentences]))
            self.vocab.append("uuunkkk")
            self.vectors = [numpy.zeros(EMBEDDING_DIM) for w in self.vocab]
    
    #Creates vectors from file or initialized to 0's    
    def buildTags(self,vectors_file = "", vocab_file = ""):
        if self.with_features:
            self.addFeatures()
        
        #Build all the tags
        self.tags = set([w[1] for w in self.sentences])
        for i,t in enumerate(self.tags):
            self.t2i[t] = i
            self.i2t[i] = t
    
    #Index all the words in the vocabluary
    def indexVocab(self):
        for i,w in enumerate(self.vocab):
            self.w2i[w] = i
            self.i2w[i] = w   
    
    def getWordIndex(self, w):
        w = w.lower()
        if w not in self.w2i:
            w = 'uuunkkk'
        return self.w2i[w]
    
    def addFeatures(self):
        new_feats = {}
        for w in self.vocab:
            new_feats["$" + w[:3]] = True
            new_feats[w[-3:] + "$"] = True

        for new in new_feats.keys():
            self.vocab.append(new)
            self.vectors.append(numpy.zeros(EMBEDDING_DIM))
            
    def makeNGram(self, mainTagger, x):
        if self.with_features:
            addFeatures = map(lambda w: ["$" + w[:3], w, w[-3:] + "$"], x)
            x = functools.reduce(lambda a, b: a + b, addFeatures, [])
        
        return [mainTagger.getWordIndex(str(w)) for w in x]
        
    def predictTest(self, model, mainTagger, out_file_name):
        file = []

        EOS = ["</s>", "</s>"]

        for i in range(0, len(self.ngrams), MINIBATCH):
            batch = self.ngrams[i:i + MINIBATCH]
            batchLength = len(batch)
            while len(batch) < MINIBATCH:
                batch.append(batch[-1])

            batchInput = [self.makeNGram(mainTagger, x) for x, y in batch]
            contextVar = torch.autograd.Variable(torch.LongTensor(batchInput))
            forward = model(contextVar)
            _, prediction = torch.max(forward, -1)

            for j, y in enumerate(map(int, prediction[:batchLength])):
                tag = mainTagger.i2t[y]
                file.append(self.ngrams[i + j][0][self.middle] + " " + tag)
                if self.ngrams[i + j][0][-self.middle:] == EOS:
                    file.append('')

        qf = open(out_file_name, 'w')
        qf.write('\n'.join(file))
        qf.close()    
        
    def epoch(self, model, loss_func, mainTagger, extra=None):
        shuffle(self.ngrams)
        count_total = 0
        count_good = 0
        total_loss = torch.Tensor([0])
        
        for i in range(0, len(self.ngrams) - MINIBATCH, MINIBATCH):
            batch = self.ngrams[i:i + MINIBATCH]
            batchInput = [self.makeNGram(mainTagger, x) for x, y in batch]
            batchOutput = [mainTagger.t2i[y] for x, y in batch]

            contextVar = torch.autograd.Variable(torch.LongTensor(batchInput))
            contextOutput = torch.autograd.Variable(torch.LongTensor(batchOutput))
            model.zero_grad()

            forward = model(contextVar)
            _, prediction = torch.max(forward, -1)
            
            if type == "pos":
                count_total += MINIBATCH
                count_good += len([1 for i, y in enumerate(map(int, prediction)) if y == batchOutput[i]])
            else:
                for i,y in enumerate(map(int, prediction)):
                    a=batch
                    b = self.i2t
                    if batch[i] != 'O' or self.i2t[y] != 'O':
                        count_total += 1
                        if y == batchOutput[i]:
                            count_good += 1 
    
            loss = loss_func(forward, contextOutput)
            total_loss += loss.data

            if extra:
                extra(loss)
        
        return round(float(count_good) / count_total, 3), round(total_loss.item() / MINIBATCH, 3)
        
def writeGraphs(out_dir, epochsData):
    files = {
        "accuracy_train": 0,
        "accuracy_dev": 1,
        "loss_train": 2,
        "loss_dev": 3
    }

    for file, i in files.items():
        pyplot.figure(i)
        pyplot.plot(range(len(epochsData)), [a[i] for a in epochsData])
        pyplot.xlabel('Epochs')
        pyplot.ylabel(file)
        pyplot.savefig(out_dir + '/' + file + '.png')
        

class NGramModeler(torch.nn.Module):
    def __init__(self, vocab_size, tags_size, with_feature):
        super(NGramModeler, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embeddings.shape = torch.Tensor(MINIBATCH, NGRAM * EMBEDDING_DIM)
        self.linear1 = torch.nn.Linear(NGRAM * EMBEDDING_DIM, HDIM)
        self.dropout1 = torch.nn.Dropout()
        self.linear2 = torch.nn.Linear(HDIM, tags_size)
        self.dropout2 = torch.nn.Dropout()
        self.with_features = with_features
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        if self.with_features:
            input_size = inputs.size()[0]
            embeds = embeds.view(input_size, NGRAM, 3, -1).view(input_size, 3, -1).sum(dim=1).view(input_size, -1)
        else:
            embeds = embeds.view(self.embeddings.shape.size())
        out = torch.tanh(self.linear1(self.dropout1(embeds)))
        out = self.linear2(self.dropout2(out))
        return out   
    
if __name__ == "__main__":
    
    dir = sys.argv[1]
    type = sys.argv[2]
    with_features = (sys.argv[3] == "y") if len(sys.argv) > 3 else False
    vectors_file = sys.argv[4] if len(sys.argv) > 4 else False
    vocab_file = sys.argv[5] if len(sys.argv) > 5 else False
    """
    dir = "./"
    type = "pos"
    with_features = False#True
    vectors_file = False#wordVectors.txt"
    vocab_file = False#"vocab.txt"
    """
    
    NGRAM = 5
    MINIBATCH = 1000
    EMBEDDING_DIM = 50
    HDIM = 128
    LR = 0.001
    EPOCHNUM = 8
    
    sufix = ""
    if with_features:
        sufix += "_f"
    if vectors_file:
        sufix += "_emb"
    out_dir = type + sufix
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    train = mainTagger(dir + type + "/train", with_features,vectors_file,vocab_file)
    if vectors_file:
        train.buildTags(vectors_file,vocab_file)
    else:    
        train.buildTags()
    train.indexVocab()
    dev = mainTagger(dir + type + "/dev", with_features)
    test = mainTagger(dir + type + "/test", with_features)
    
    model = NGramModeler(len(train.vocab), len(train.tags), with_features)
    model.embeddings.weight.data.copy_(torch.from_numpy(numpy.stack(map(lambda a: numpy.array(a), train.vectors))))
    vects = train.vectors
    
    
    epochsData = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_func = torch.nn.CrossEntropyLoss()    
    def backwards(loss):
        loss.backward()
        optimizer.step()
    
    for epoch in range(EPOCHNUM):
        accuracy, loss = train.epoch(model, loss_func, train, backwards)
        print("epoch", epoch, '\tloss', loss, '\taccuracy', accuracy, '\t')
        d_accuracy, d_loss = dev.epoch(model, loss_func, train)       
        print("dev", '\t\tloss', d_loss, '\taccuracy', d_accuracy, '\t')
        epochsData.append([accuracy, d_accuracy, loss, d_loss])       
        test.predictTest(model, train, out_dir + "/test1." + type)
        writeGraphs(out_dir, epochsData)
 
    