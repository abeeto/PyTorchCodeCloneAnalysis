import time
import torch.nn.functional as F

from utils import *
from discriminator import *
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pretrainDiscriminator(train_p, train_h, train_l, 
                        disc, disc_optimizer, criterion):
    """
    Core training function
    """
    train_l = torch.tensor(train_l, device=device).view(-1)
    disc_hidden = disc.initHidden()
    disc_optimizer.zero_grad()

    _, _, _, _, _, y_ = disc(train_p, train_h)
    # print 'y_: ', y_, 'label: ', train_l
    loss = criterion(y_.view(1,-1), train_l)

    loss.backward()
    return loss

def pretrainItersDisc(disc, phpairs, epochs=1, mini_batch=32, learning_rate=0.01):
    loss = 0.
    print_every = mini_batch

    disc_optim = optim.SGD(disc.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for i in range(0, len(phpairs), mini_batch):
            batch_data = phpairs[i: i+mini_batch]
            for datum in batch_data:
                loss += pretrainDiscriminator(datum.premise, datum.hypothesis, 
                                        datum.label, disc, disc_optim, criterion)
            disc_optim.step()
            # print list(disc.parameters())[1].grad
            if i % print_every == 0:
                print "train data %d/%d, loss:%.4f\n" % (i, len(phpairs), loss/print_every)
                loss = 0.
                # checkGrad(disc)
    

def checkGrad(model):
    """
    Check whether the model parameters have gradient properly
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print '%s has proper grad' % name
        else:
            print '%s doesn\'t have grad' % name
