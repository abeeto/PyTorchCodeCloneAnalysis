from model import Model
from config import Config
from utils import build_data, load_vocab, get_processing_word, Dataset,\
                clear_data_path, get_trimmed_glove_vectors,Dataset, write_clear_data_pd
import sys
with open(sys.argv[1], "r") as f:
    pipeline = '\t'.join(f.readlines())

config = Config(pipeline)
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

test_filepath, _ = write_clear_data_pd(
    config.test_filename, config.DEFAULT, domain=config.domain)
test = Dataset(test_filepath, processing_word, processing_label,
               config.max_iter)

# build model
model = Model(
    config, embeddings, ntags=len(vocab_labels), nchars=len(vocab_chars))
# build graph
model.build_graph()
model.evaluate(test, vocab_labels)

# processing_word = get_processing_word(
#     vocab_words, vocab_chars, lowercase=True, chars=config.chars)
# tags = load_vocab(config.labels_filename)
# idx_to_tag = {idx: tag for tag, idx in vocab_labels.items()}
# saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, self.config.model_output)

# import tensorflow as tf

# config = Config()
# tags = load_vocab(config.labels_filename)
# # saver = tf.train.Saver()
# saver = tf.train.import_meta_graph(config.model_output + "best.meta")
# # logits = tf.get_collection("logits")[0]
# graph = tf.get_default_graph()
# logits = graph.get_tensor_by_name("logits:0")
# # transition_params = tf.get_collection('transition_params')
# with tf.Session() as sess:
#     sess.run(logits)

# if config.crf:
#     log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
#         logits, self.labels, self.sequence_lengths)
#     self.loss = tf.reduce_mean(-log_likelihood)
# else:
#     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=self.logits, labels=self.labels)
#     mask = tf.sequence_mask(self.sequence_lengths)
#     losses = tf.boolean_mask(losses, mask)
#     self.loss = tf.reduce_mean(losses)

# with tf.Session() as sess:
#     saver.restore(sess, config.model_output + "best")
#     acc, f1 = self.run_evaluate(sess, test, tags)
#     self.logger.info(
#         "- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
