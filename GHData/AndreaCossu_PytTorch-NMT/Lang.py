class Lang:
    def __init__(self, name):
        # Language initialization
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Counting SOS, EOS and UNK

    def index_words(self, sentence):
        # Sentence indexing
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        # Indexing of (possibly new) word
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
