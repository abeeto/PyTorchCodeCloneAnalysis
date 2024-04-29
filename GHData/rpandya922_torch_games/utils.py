import torch as th
from torch import nn

def bach_partner(obs):
    return 0

def stravinsky_partner(obs):
    return 1

def helpful_partner(obs):
    # always picks what the robot picked last time
    return obs[0]

def adversarial_partner(obs):
    # always picks the opposite of what the robot picked last time
    if obs[0] == 1:
        return 0
    return 1


def train(model, optimizer, trainset_loader, valset_loader, epoch=50, loss_fn=nn.MSELoss):
    all_train_loss = []
    all_val_loss = []
    batch_size = trainset_loader.batch_size

    iteration = 0
    for ep in range(epoch):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # TODO: remove once LSTM training is debugged
            data = data.swapaxes(0, 1)
            target = target.squeeze(1).type(th.LongTensor) # TODO: change to accept either categorical or float target
            # forward pass
            model_out = model(data)

            # compute loss
            # TODO: understand why CE loss broke things
            # loss = nn.MSELoss(reduction="sum")
            # loss = loss_fn(reduction="sum")
            loss = loss_fn()
            output = loss(model_out, target)
            total_loss += output.item()

            optimizer.zero_grad()
            output.backward()
            optimizer.step()

            iteration += 1
        all_train_loss.append(total_loss / (batch_idx+1) / batch_size)

        # test on validation
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(valset_loader):
            data = data.swapaxes(0, 1)
            target = target.squeeze(1).type(th.LongTensor)

            model_out = model(data)

            # loss = nn.MSELoss(reduction="sum")
            loss = loss_fn()
            val_loss += loss(model_out, target).item()
        all_val_loss.append(val_loss / (batch_idx+1) / batch_size)

    return all_train_loss, all_val_loss