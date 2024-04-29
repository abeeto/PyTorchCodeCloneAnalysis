import os
import sys
import time
import torch
import argparse

from model import SASRec
from utils import *
from data_iterator import *

best_metric = 0

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--topN', default=10, type=int)
parser.add_argument('--num_interest', default=4, type=int)
parser.add_argument('--test_iter', default=10, type=int)
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')


def train(train_file, valid_file, test_file, cate_file, item_count, dataset = "book", batch_size = 128,
        maxlen = 100, test_iter = 50, model_type = 'DNN', lr = 0.001, max_iter = 100, patience = 20):

    item_cate_map = load_item_cate(cate_file)

    # (user_id_list, item_id_list), (hist_item_list, hist_mask_list)
    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)

    model = SASRec(item_count, args).to(args.device)
    model.train()  # enable model training

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    print('training begin')
    sys.stdout.flush()

    start_time = time.time()
    try:
        trials = 0
        iter = 0
        # train_data: userid, itemid, sql_num
        for src, tgt in train_data:
            if iter % test_iter == 0:
                print('The', iter / test_iter + 1, 'test_iter:')
            iter += 1
            """
            训练
            """
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
            nick_id, item_id, hist_item, hist_mask = np.array(nick_id), np.array(item_id), np.array(hist_item), np.array(hist_mask)
            model(hist_item)
            user_eb = model.output_user(hist_item, item_id, hist_mask)
            item_embs = model.output_item()
            adam_optimizer.zero_grad()
            # 负采样
            loss = sample_softmax_loss(item_embs, item_id, item_count, user_eb, hist_item)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

            if iter % test_iter == 0:
                print('test_iter:', iter / test_iter, 'loss: ', loss.item())
                model.eval()
                metrics = evaluate_full(valid_data, model, args, item_cate_map)
                log_str = ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(log_str)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    global best_metric
                    if recall > best_metric:
                        best_metric = recall
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            break

                test_time = time.time()
                print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                sys.stdout.flush()
                model.train()

            if iter >= max_iter:
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':

    print(sys.argv)
    args = parser.parse_args()

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'ml-1m':
        path = '/content/SASRecPyTorch/data/ml-1m_data/'
        # path = './data/ml-1m_data/'
        item_count = 3417
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter
    elif args.dataset == 'book':
        path = '/content/SASRecPyTorch/data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    elif args.dataset == 'ml-10m':
        path = '/content/SASRecPyTorch/data/ml-10m_data/'
        item_count = 10197
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file,
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter,
              model_type=args.model_type, lr=args.lr, max_iter=args.max_iter, patience=args.patience)
    else:
        print('do nothing...')