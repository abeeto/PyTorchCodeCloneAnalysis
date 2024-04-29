import options
import argparse
import evaluate
import models
import random
import func
import utils
import optimization
from torch.nn.utils import clip_grad_norm_


def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    options.data_opts(parser)
    return parser.parse_args()


def run_epoch(opt, model, feeder, optimizer, batches):
    model.train()
    nbatch = 0
    criterion = models.make_loss_compute()
    while nbatch < batches:
        _, cs, qs, chs, qhs, y1s, y2s, ct, qt = feeder.next(opt.batch_size)
        nbatch += 1
        logits1, logits2 = model(func.tensor(cs), func.tensor(qs), func.tensor(chs), func.tensor(qhs), ct, qt)
        t1, t2 = func.tensor(y1s), func.tensor(y2s)
        loss = (criterion(logits1, t1) + criterion(logits2, t2)).mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        optimizer.step()
        print('------ITERATION {}, {}/{}, epoch: {:>.2F}% loss: {:>.4F}'.format(feeder.iteration, feeder.cursor, feeder.size, 100.0*nbatch/batches, loss.tolist()))


class Logger(object):
    def __init__(self, opt):
        self.output_file = opt.summary_file
        self.lines = list(utils.read_all_lines(self.output_file))


    def __call__(self, message):
        print(message)
        self.lines.append(message)
        utils.write_all_lines(self.output_file, self.lines)


def train(steps=400, evaluate_size=None):
    func.use_last_gpu()
    opt = make_options()
    model, optimizer, feeder, ckpt = models.load_or_create_models(opt, True)
    autodecay = optimization.AutoDecay(optimizer, patience=30, max_lr=opt.learning_rate)
    log = Logger(opt)
    if ckpt is not None:
        _, last_accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size, char_limit=opt.char_limit, size=evaluate_size)
    else:
        last_accuracy = 0
    while True:
        run_epoch(opt, model, feeder, optimizer, steps)
        em, accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.validate_batch_size, char_limit=opt.char_limit, size=evaluate_size)
        if accuracy > last_accuracy:
            models.save_models(opt, model, optimizer, feeder)
            last_accuracy = accuracy
            autodecay.better()
            log('MODEL SAVED WITH ACCURACY EM:{:>.2F}, F1:{:>.2F}.'.format(em, accuracy))
        else:
            autodecay.worse()
            if autodecay.should_stop():
                models.restore(opt, model, optimizer, feeder)
                autodecay = optimization.AutoDecay(optimizer, max_lr=opt.learning_rate)
                log(f'MODEL RESTORED {accuracy:>.2F}/{last_accuracy:>.2F}, decay = {autodecay.decay_counter}, lr = {autodecay.learning_rate:>.6F}.')
            else:
                log(f'CONTINUE TRAINING {accuracy:>.2F}/{last_accuracy:>.2F}, decay = {autodecay.decay_counter}, lr = {autodecay.learning_rate:>.6F}.')
train()