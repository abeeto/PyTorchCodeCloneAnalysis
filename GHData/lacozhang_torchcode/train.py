from model import Model
from config import Config
from utils import build_data, load_vocab, get_processing_word, Dataset,\
                clear_data_path, get_trimmed_glove_vectors,Dataset
import sys
with open(sys.argv[1], "r") as f:
    pipeline = '\t'.join(f.readlines())
config = Config(pipeline)

if not config.reload:
    build_data(config)

# load vocabs
vocab_words = load_vocab(config.words_filename)
vocab_labels = load_vocab(config.labels_filename)
vocab_chars = load_vocab(config.chars_filename)

# get processing functions
processing_word = get_processing_word(
    vocab_words, vocab_chars, lowercase=True, chars=config.chars)
processing_label = get_processing_word(
    vocab_labels, lowercase=False, label_vocab=True)

# get pre trained embeddings
embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

# create dataset
dev = Dataset(
    clear_data_path(config.dev_filename), processing_word, processing_label,
    config.max_iter)
test = Dataset(
    clear_data_path(config.test_filename), processing_word, processing_label,
    config.max_iter)
train = Dataset(
    clear_data_path(config.train_filename), processing_word, processing_label,
    config.max_iter)

# build model
model = Model(
    config, embeddings, ntags=len(vocab_labels), nchars=len(vocab_chars))
# build graph
model.build_graph()

# train, evaluate and interact
model.train(train, dev, vocab_labels)
model.evaluate(test, vocab_labels)