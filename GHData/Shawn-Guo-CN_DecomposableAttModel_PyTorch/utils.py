import os
from argparse import ArgumentParser
import logging
import logging.config
import json
import time


def get_args():
    parser = ArgumentParser(description='Batched SNLI with Decomposable Model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=200)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--d_F', type=int, default=200)
    parser.add_argument('--d_G', type=int, default=200)
    parser.add_argument('--d_H', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--dp_ratio', type=float, default=0.2)
    parser.add_argument('--use-realdata', action='store_false', dest='sample_data')
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--snli_root', type=str, default='/data/shawnguo/')
    parser.add_argument('--data_cache', type=str, default=os.path.join('/data/shawnguo/', 'GloVe'))
    parser.add_argument('--vector_cache', type=str,
                        default=os.path.join('/data/shawnguo/', 'GloVe/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--snapshot_prefix', type=str, default='')
    parser.add_argument('--only_test', action='store_true', dest='test')
    args = parser.parse_args()
    return args


def setup_logging(
        log_path_prefix='log/',
        default_path='logging.json',
        default_level=logging.DEBUG,
        env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            handlers = config['handlers']
            log_folder = log_path_prefix + str(time.localtime()[1]) + '_' + str(time.localtime()[2]) + \
                         '_' + str(time.localtime()[3]) + '_' + str(time.localtime()[4]) + '/'
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            handlers['debug_file_handler']['filename'] = os.path.join(log_folder, 'debug.log')
            handlers['info_file_handler']['filename'] = os.path.join(log_folder, 'info.log')
            handlers['error_file_handler']['filename'] = os.path.join(log_folder, 'error.log')
            config['handlers'] = handlers
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
