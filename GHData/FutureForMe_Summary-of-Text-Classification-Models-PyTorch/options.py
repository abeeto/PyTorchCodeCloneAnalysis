import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./results')

parser.add_argument('-model_name', type=str, default='TextRCNN', choices=['MLP', 'TextCNN', 'TextRNN', 'Bert'])
parser.add_argument('-data_name', type=str, default='20news', choices=['sogou_news', '20news', 'THUCNews'])
parser.add_argument('-train_path', type=str, default='./data/THUCNews/train.txt')
parser.add_argument('-dev_path', type=str, default='./data/THUCNews/dev.txt')
parser.add_argument('-test_path', type=str, default='./data/THUCNews/test.txt')
parser.add_argument('-vocab_path', type=str, default='./data/THUCNews/vocab.pkl')
parser.add_argument('-use_word', type=bool, default=False)

# Hyper parameters
parser.add_argument('-pad_size', type=int, default=32)
parser.add_argument('-batch_size', type=float, default=128)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-epoch_nums', type=int, default=50)
parser.add_argument('-embed_size', type=int, default=300)
parser.add_argument('-vocab_nums', type=int, default=0)
parser.add_argument('-patience', type=int, default=10)
parser.add_argument('-class_nums', type=int, default=10)
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-test_model', type=str, default=None)
parser.add_argument('-gpu', type=int, default=0, help='if >=0 use gpu if =-1 use cpu')
parser.add_argument('-criterion', type=str, default='f1_micro', choices=['f1_micro', 'loss_dev'])

args = parser.parse_known_args()

# model parameters
if args[0].model_name == 'MLP':              # MLP
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-input_dim', type=int, default=32)
    parser.add_argument('-hidden_dim', type=int, default=128)

elif args[0].model_name == 'TextRNN':        # TextRNN
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-hidden_dim', type=int, default=128)

elif args[0].model_name == 'TextCNN':        # TextCNN
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-filter_size', type=str, default='2, 3, 4')
    parser.add_argument('-filter_map_nums', type=int, default=256)

elif args[0].model_name == 'TextRCNN':        # TextRCNN
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-hidden_dim', type=int, default=512)

elif args[0].model_name == 'Bert':
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-hidden_dim', type=int, default=768)
    parser.add_argument('-pretrain_path', type=str, default='/home/Disk/wanghaotian/Pretrian_Model/bert_base_chinese')

args = parser.parse_args()
