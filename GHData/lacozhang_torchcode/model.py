import numpy as np
import tensorflow as tf
from utils import pad_sequences, Progbar, get_chunks, minibatches, load_vocab, infer_eval
import os
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model(object):
    def __init__(self, config, embeddings, ntags, nchars=None):

        self.config = config
        self.nchars = nchars
        self.ntags = ntags
        self.embeddings = embeddings
        self.logger = config.logger  # now instantiated in config
        self.vocab_words = load_vocab(self.config.words_filename)
        self.vocab_labels = load_vocab(self.config.labels_filename)
        self.idx_to_word = {
            idx: word
            for word, idx in self.vocab_words.items()
        }
        self.idx_to_tag = {
            idx: word
            for word, idx in self.vocab_labels.items()
        }

    # def build_vocabs(self):
    #     self.nchars = nchars
    #     self.ntags = ntags
    #     pass

    def add_placeholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(
            tf.int32, shape=[None, None], name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(
            tf.int32, shape=[None], name="sequence_lengths")
        # shape = (batch size, max length of sentence, max length of words)
        self.char_ids = tf.placeholder(
            tf.int32, shape=[None, None, None], name="char_ids")
        # shape = (batch size, max length of sentence in batch)
        self.word_lengths = tf.placeholder(
            tf.int32, shape=[None, None], name="word_lengths")
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(
            tf.int32, shape=[None, None], name="labels")
        # hyper parameters
        self.dropout = tf.placeholder(tf.float32, shape=[], name="drop_out")
        self.LR = tf.placeholder(tf.float32, shape=[], name="LR")

    def get_feed_dict(self, words, labels=None, LR=None, dropout=None):
        if self.config.chars:
            char_ids, word_ids = zip(*words)  # ==========important
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
            char_ids, word_lengths = pad_sequences(
                char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, pad_tok=0)

        feed_dict = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if self.config.chars:
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_lengths] = word_lengths
        if labels is not None:
            labels, _ = pad_sequences(labels, pad_tok=0)
            feed_dict[self.labels] = labels
        if LR is not None:
            feed_dict[self.LR] = LR
        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, sequence_lengths

    def add_word_embeddings_op(self):
        with tf.variable_scope("embed_words"):
            _word_embeddings = tf.Variable(
                self.embeddings,
                name="import_embeddings",
                dtype=tf.float32,
                trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(
                _word_embeddings, self.word_ids, name="word_embeddings")

        if self.config.chars:
            with tf.variable_scope("embed_chars"):
                _char_embeddings = tf.get_variable(
                    "init_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(
                    _char_embeddings, self.char_ids, name="char_embeddings")
                # shape = (batch size, max length of sentences, max length of words, dim_chars)
                s = tf.shape(char_embeddings)
                # shape = (time batch size, max length of words, dim_chars)
                char_embeddings = tf.reshape(
                    char_embeddings, shape=[-1, s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])
                cell_fw_char = tf.contrib.rnn.LSTMCell(
                    self.config.char_hidden_size)
                cell_bw_char = tf.contrib.rnn.LSTMCell(
                    self.config.char_hidden_size)

                _, ((_, output_fw_char),
                    (_, output_bw_char)) = tf.nn.bidirectional_dynamic_rnn(
                        cell_bw_char,
                        cell_fw_char,
                        char_embeddings,
                        sequence_length=word_lengths,
                        dtype=tf.float32)
                # shape = (time batch size, 1, 2 * char_hidden_size)
                output = tf.concat([output_fw_char, output_bw_char], axis=-1)
                # shape = (time batch size, max length of sentences, 2 * char_hidden_size)
                output = tf.reshape(
                    output, shape=[-1, s[1], 2 * self.config.char_hidden_size])
                # shape = (time batch size, max length of sentences, 2 * char_hidden_size+embeddings vector size)
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        with tf.variable_scope("bi-LSTM"):
            cell_fw_word = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            cell_bw_word = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw_word,
             output_bw_word), _ = tf.nn.bidirectional_dynamic_rnn(
                 cell_fw_word,
                 cell_bw_word,
                 self.word_embeddings,
                 sequence_length=self.sequence_lengths,
                 dtype=tf.float32)
            # shape = (time batch size, max length of sentences, 2 * hidden_size)
            output = tf.concat([output_fw_word, output_bw_word], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("logits"):
            W = tf.get_variable(
                "W",
                shape=[2 * self.config.hidden_size, self.ntags],
                dtype=tf.float32)

            b = tf.get_variable(
                "b",
                shape=[self.ntags],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            ntimes_steps = tf.shape(output)[1]
            # shape = (time batch size * max length of sentences, 2 * hidden_size)
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size])
            # shape = (time batch size * max length of sentences, ntags)
            pred = tf.matmul(output, W) + b
            # shape = (time batch size, max length of sentences, ntags)
            self.logits = tf.reshape(
                pred, [-1, ntimes_steps, self.ntags], name="logits")

    def add_loss_op(self):
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            # pick up real words' losses
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def add_pred_op(self):
        if not self.config.crf:
            self.labels_pred = tf.cast(
                tf.argmax(self.logits, axis=-1), tf.int32, name="labels_pred")

    def add_train_op(self):
        with tf.variable_scope("train_op"):
            # sgd method
            if self.config.LR_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.LR)
            elif self.config.LR_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.LR)
            elif self.config.LR_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.LR)
            elif self.config.LR_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.LR)
            else:
                raise NotImplementedError(
                    "Unknown train op {}".format(self.config.LR_method))
            # gradient clipping if config.clip is positive
            if self.config.clip > 0:
                gradients, variables = zip(
                    *optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(
                    gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(
                    zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss, name="train_op")

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 sess.graph)

    def build_graph(self):
        # self.build_vocabs()
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def predict_batch(self, sess, words):
        feed, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run(
                [self.logits, self.transition_params], feed_dict=feed)
            for logit, sequence_length in zip(logits, sequence_lengths):
                # get real word infor in each batch
                score = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                    score, transition_params)
                viterbi_sequences += [viterbi_sequence]

            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=feed)
            return labels_pred, sequence_lengths

    def run_epoch(self, sess, train, dev, tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
            epoch: (int) number of the epoch
        """
        nbatches = (
            len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, labels
                ) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.LR,
                                       self.config.dropout)

            _, train_loss, summary = sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev, tags)
        self.logger.info(
            "- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, f1

    def idx_restore(self, word, lab, lab_pred):
        res = ""
        for w, l, lp in zip(word, lab, lab_pred):
            res += self.idx_to_word[w] + "\t"
            res += self.idx_to_tag[l] + "\t"
            res += self.idx_to_tag[lp] + "\n"
        res += "\n"
        return res

    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags, self.config.DEFAULT))
                lab_pred_chunks = set(
                    get_chunks(lab_pred, tags, self.config.DEFAULT))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1

    def run_infer(self, sess, test, tags):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """
        infer_res = open(self.config.infer_filename, 'w', encoding="utf-8-sig")
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            words_copy = copy.deepcopy(words)
            labels_pred, sequence_lengths = self.predict_batch(sess, words)
            # print("predict_batch", labels_pred, sequence_lengths,words_copy)
            if self.config.chars:
                _, words_res = zip(*words_copy)
            else:
                words_res = words_copy
            for word_res, lab, lab_pred, length in zip(
                    words_res, labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                # print("idx_restore", word_res, lab, lab_pred)
                infer_res.write(self.idx_restore(word_res, lab, lab_pred))
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags, self.config.DEFAULT))
                lab_pred_chunks = set(
                    get_chunks(lab_pred, tags, self.config.DEFAULT))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
        infer_res.close()
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1

    def train(self, train, dev, tags):
        """
        Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
        """
        best_score = 0
        saver = tf.train.Saver(max_to_keep=self.config.max_model_to_keep)
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)
            if self.config.reload:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.config.model_output))
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(
                    epoch + 1, self.config.nepochs))

                ACC, F1 = self.run_epoch(sess, train, dev, tags, epoch)
                self.config.LR *= self.config.LR_decay

                if F1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output + "best")
                    best_score = F1
                    self.logger.info("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info(
                            "- early stopping {} epochs without improvement".
                            format(nepoch_no_imprv))
                        break
        # tf.add_to_collection("transition_params", self.transition_params)

    def evaluate(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess,
                          tf.train.latest_checkpoint(self.config.model_output))
            ACC, F1 = self.run_infer(sess, test, tags)
            # self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(
            #     100 * ACC, 100 * F1))
            eval_path = infer_eval(self.config.infer_filename)
            self.logger.info("finish eval -> {}".format(eval_path))
