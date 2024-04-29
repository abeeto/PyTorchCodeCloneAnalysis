import glob

import torch
import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from utils import *
from model import *

logger = logging.getLogger('mylogger')
log_path_prefix = 'log/'
setup_logging(log_path_prefix)

args = get_args()
args.cuda = torch.cuda.is_available()

if args.cuda:
    torch.cuda.set_device(args.gpu)
else:
    logging.error('CUDA is not available')
    exit(-1)

inputs = data.Field(lower=args.lower)
answers = data.Field(sequential=False)

print 'loading and spliting SNLI...',
if args.sample_data:
    train, dev, test = datasets.SNLI.splits(inputs, answers, root=args.snli_root, train='train_sample.jsonl',
                                            validation='dev_sample.jsonl', test='test_sample.jsonl')
else:
    train, dev, test = datasets.SNLI.splits(inputs, answers, root=args.snli_root, train='train.jsonl',
                                            validation='dev.jsonl', test='test.jsonl')
print 'done'

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        logger.info('loading word embeddings from cache file' + args.vector_cache)
        inputs.vocab.vectors = torch.load(args.vector_cache)
        logger.info('done')
    else:
        logger.info('loading word embeddings from raw file' + args.data_cache + args.word_vectors + '.txt')
        inputs.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
        if not os.path.exists(os.path.dirname(args.vector_cache)):
            os.makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
        logger.info('done')
answers.build_vocab(train)
answers.vocab.itos = [u'entailment', u'contradiction', u'neutral']
answers.vocab.stoi = {u'entailment': 0, u'contradiction': 1, u'neutral': 2}

if args.cuda:
    args.batch_size = 32
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=args.gpu)
else:
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=-1)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

logger.info('building model ...', )
model = DecomposableModel(config)
if args.word_vectors:
    model.embed.weight.data = inputs.vocab.vectors
logger.info('done')
best_dev_acc = -1

if args.resume_snapshot:
    logger.info('rebuilding model from ' + args.resume_snapshot + '...', )
    model_state_dict = torch.load(args.resume_snapshot)
    model.load_state_dict(model_state_dict)
    del model_state_dict
    logger.info('done')
    best_dev_acc = float(args.resume_snapshot.split('\\')[-1].split('_')[3])

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.test:
    logger.info('entered test mode')
    model.eval()
    test_iter.init_epoch()
    n_test_correct, test_loss = 0, 0
    for test_batch_idx, test_batch in enumerate(test_iter):
        answer = model(test_batch)
        n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
        test_loss = criterion(answer, test_batch.label)
    test_acc = 100. * n_test_correct / len(test)
    logger.info('test acc: %.2f' % test_acc)
    exit(0)

iterations = 0
start = time.time()
train_iter.repeat = False
header = '   Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template  = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template  =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
logger.info(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        model.train()
        optimizer.zero_grad()
        iterations += 1
        answer = model(batch)
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        loss = criterion(answer, batch.label)
        loss.backward()
        optimizer.step()

        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            torch.save(model.state_dict(), snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)
        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                 answer = model(dev_batch)
                 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                 dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)
            logger.info(
                dev_log_template.format(time.time() - start,
                                        epoch, iterations, 1 + batch_idx, len(train_iter),
                                        100. * (1 + batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0],
                                        train_acc, dev_acc)
            )
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, args.snapshot_prefix + 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
        elif iterations % args.log_every == 0:
            logger.info(
                log_template.format(time.time() - start,
                                    epoch, iterations, 1 + batch_idx, len(train_iter),
                                    100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8, train_acc,
                                    ' ' * 12)
            )

