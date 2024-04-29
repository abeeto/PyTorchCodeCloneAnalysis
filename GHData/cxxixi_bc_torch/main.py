"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
# import chainer
import torch
import opts
import models
import dataset
from models.envnet import EnvNet
from train import Trainer


def main():
    opt = opts.parse()
    for split in opt.splits:
        print('+-- Split {} --+'.format(split))
        train(opt, split)


def train(opt, split):

    ## set up optimizer, models and enviroment##

    model = getattr(models, opt.netType)(opt.nClasses)
    model = model.cuda()
    # model = EnvNet(5)
    # pp=0
    # for p in list(model.parameters()):
    #     nnd=1
    #     for s in list(p.size()):
    #         nnd = nnd*s
    #     pp += nnd
    # print(pp)
    # model.cuda()????

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, weight_decay=opt.weightDecay, nesterov=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    ## train and val dataloader ##
    trainloader, valloader = dataset.setup(opt, split)

    # run_training(model, optimizer, trainloader, valloader, opt)
    trainer = Trainer(model, optimizer, trainloader, valloader, opt)

    # ## if we only want to test ##.  ??????????????
    # if opt.testOnly:
    #     chainer.serializers.load_npz(
    #         os.path.join(opt.save, 'model_split{}.npz'.format(split)), trainer.model)
    #     val_top1 = trainer.val()
    #     print('| Val: top1 {:.2f}'.format(val_top1))
    #     return



    ## trainin with given epoch ##
    for epoch in range(1, opt.nEpochs + 1):


        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} |  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, train_loss, train_top1, val_top1))
        sys.stdout.flush()

    ## save models ##
    if opt.save != 'None':
        chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_split{}.npz'.format(split)), model)


if __name__ == '__main__':
    main()
