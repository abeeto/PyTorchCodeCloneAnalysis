import argparse

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-char_hidden_size', type=int, default=0)
    group.add_argument('-encoder_hidden_size', type=int, default=75)
    group.add_argument('-rnn_type', type=str, default='gru')
    group.add_argument('-ckpt_path', type=str, default='./checkpoint/model.pt')
    group.add_argument('-with_elmo', type=int, default=1)
    group.add_argument('-embedding_dim', type=int, default=512)


def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-batch_size', type=int, default=64)
    group.add_argument('-validate_batch_size', type=int, default=64)
    group.add_argument('-learning_rate', type=float, default=0.002)
    group.add_argument('-dropout', type=float, default=0.3)
    group.add_argument('-max_grad_norm', type=float, default=5.0)
    group.add_argument('-summary_file', type=str, default='./output/summary.txt')


def evaluate_opts(parser):
    group = parser.add_argument_group('evaluate')
    group.add_argument('-batch_size', type=int, default=64)
    group.add_argument('-output_file', type=str, default='./output/evaluate.txt')
    group.add_argument('-dropout', type=float, default=0)


def preprocess_opts(parser):
    group = parser.add_argument_group('preprocess')
    group.add_argument('-squad_train_file', type=str, default='./data/squad/train-v1.1.json')
    group.add_argument('-squad_dev_file', type=str, default='./data/squad/dev-v1.1.json')
    group.add_argument('-squad_test_file', type=str, default='./data/squad/dev-v1.1.json')
    group.add_argument('-drcd_train_file', type=str, default='./data/drcd/DRCD_training.json')
    group.add_argument('-drcd_dev_file', type=str, default='./data/drcd/DRCD_dev.json')
    group.add_argument('-drcd_test_file', type=str, default='./data/drcd/DRCD_dev.json')
    group.add_argument('-cmrc_train_file', type=str, default='./data/cmrc/cmrc2018_train.json')
    group.add_argument('-cmrc_dev_file', type=str, default='./data/cmrc/cmrc2018_dev.json')
    group.add_argument('-cmrc_test_file', type=str, default='./data/cmrc/cmrc2018_trial.json')
    group.add_argument('-cws', type=str, default='char')
    group.add_argument('-glove_word_emb_file', type=str, default='./data/glove/glove.840B.300d.txt')


def data_opts(parser):
    group = parser.add_argument_group('data')
    group.add_argument('-word_dim', type=int, default=300)
    group.add_argument('-char_dim', type=int, default=8)
    group.add_argument('-char_limit', type=int, default=16)
    group.add_argument('-word_emb_file', type=str, default='./generate/emb.word.json')
    group.add_argument('-char_emb_file', type=str, default='./generate/emb.char.json')
    group.add_argument('-w2i_file', type=str, default='./generate/w2i.json')
    group.add_argument('-c2i_file', type=str, default='./generate/c2i.json')
    group.add_argument('-dataset', type=str, default='squad')
    group.add_argument('-meta_file', type=str, default='./generate/meta.json')
    group.add_argument('-max_passage_tokens', type=int, default=1000)

    group.add_argument('-train_example_file', type=str, default='./generate/example.train.json')
    group.add_argument('-dev_example_file', type=str, default='./generate/example.dev.json')
    group.add_argument('-test_example_file', type=str, default='./generate/example.test.json')

