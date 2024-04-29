import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

torch.manual_seed(1)


def test():
    lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
    # data is 2x5.  A maps from 5 to 3... can we map "data" under A?
    data = autograd.Variable(torch.randn(2, 5))
    print(lin(data))  # yes
    lin = lin(data)


    print(F.softmax(lin))
    print(F.softmax(lin).sum())
    print(F.log_softmax(data))


def BOW_test():
    data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
            ("Give it to me".split(), "ENGLISH"),
            ("No creo que sea una buena idea".split(), "SPANISH"),
            ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

    test_data = [("Yo creo que si".split(), "SPANISH"),
                 ("it is lost on me".split(), "ENGLISH")]


    word_to_ix = {}
    for sent, _ in data + test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)

    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = 2



    class BoWClassifier(nn.Module):  # inheriting from nn.Module!

        def __init__(self, num_labels, vocab_size):
            # calls the init function of nn.Module.  Dont get confused by syntax,
            # just always do it in an nn.Module
            super(BoWClassifier, self).__init__()

            # Define the parameters that you will need.  In this case, we need A and b,
            # the parameters of the affine mapping.
            # Torch defines nn.Linear(), which provides the affine map.
            # Make sure you understand why the input dimension is vocab_size
            # and the output is num_labels!
            self.linear = nn.Linear(vocab_size, num_labels)

            # NOTE! The non-linearity log softmax does not have parameters! So we don't need
            # to worry about that here

        def forward(self, bow_vec):
            # Pass the input through the linear layer,
            # then pass that through log_softmax.
            # Many non-linearities and other functions are in torch.nn.functional
            return F.log_softmax(self.linear(bow_vec))

    def make_bow_vector(sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)


    def make_target(label, label_to_ix):
        return torch.LongTensor([label_to_ix[label]])

    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)


    # the model knows its parameters.  The first output below is A, the second is b.
    # Whenever you assign a component to a class variable in the __init__ function
    # of a module, which was done with the line
    # self.linear = nn.Linear(...)
    # Then through some Python magic from the Pytorch devs, your module
    # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
    for param in model.parameters():
        print(param)

    # To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(autograd.Variable(bow_vector))
    print(log_probs)

    label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

    # print log probs before training.
    for instance, label in test_data:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)
        print("Old log_probs!")
        print(log_probs)
    #
    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)
    #
    #
    for i in range(100):
        for instance, label in data:
            model.zero_grad()
            bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
            target = autograd.Variable(make_target(label, label_to_ix))

            log_probs = model(bow_vec)

            loss = loss_func(log_probs, target)
            loss.backward()
            optimizer.step()
    #
    for instance, label in test_data:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)
        print('Now log_probs!')
        print(log_probs)




def word_embed_test():
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
    hello_embed = embeds(autograd.Variable(lookup_tensor))
    print(lookup_tensor)
    print(hello_embed)


def cbow_test():
    CONTEXT_SIZE = 2
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []

    # data-creation
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    print(data[:5])

    class CBOW(nn.Module):

        def __init__(self):
            pass

        def forward(self, inputs):
            pass

    # create your model and train.  here are some functions to help you make
    # the data ready for use by your module

    def make_context_vector(context, word_to_ix):
        idxs = [word_to_ix[w] for w in context]
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)

    make_context_vector(data[0][0], word_to_ix)  # example


def lstm_test():
    lstm = nn.LSTM(3, 3)
    inputs = [autograd.Variable(torch.randn(1, 3))
              for _ in range(5)]

    hidden = (autograd.Variable(torch.randn(1, 1, 3)),
              autograd.Variable(torch.randn(1, 1, 3)))

    # for i in inputs:
    #     out, hidden = lstm(i.view(1, 1, -1), hidden)

    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    out, hidden = lstm(inputs, hidden)
    print("Out!")
    print(out)
    print("Hidden!")
    print(hidden)

def lstm_tagger():

    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        # with torch.cuda.device(2)
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)

    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    VOCAB_SIZE = len(word_to_ix)
    TAGSET_SIZE = len(tag_to_ix)

    class LSTM_Tagger(nn.Module):

        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(LSTM_Tagger, self).__init__()
            self.hidden_dim = hidden_dim
            # These are all functional
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.hidden = self.hidden_init()

        def hidden_init(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))


        def forward(self, sentence):
            embeds = self.word_embedding(sentence.view(len(sentence), -1))
            # pdb.set_trace()
            lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space)
            return tag_scores



    model = LSTM_Tagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE)
        # .cuda()
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

    for i in range(300):
        for sentence, tags in training_data:
            # pdb.set_trace()
            # we need to clean the grad & hidden state each time!!
            model.zero_grad()
            model.hidden = model.hidden_init()

            inputs = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)


            tag_scores = model(inputs.view(1, 1, -1))

            loss = loss_func(tag_scores, targets)
            loss.backward()
            optimizer.step()

    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)








if __name__ == "__main__":
    # word_embed_test()
    # lstm_test()
    lstm_tagger()
