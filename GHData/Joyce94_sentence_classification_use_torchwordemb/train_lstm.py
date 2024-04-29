import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import random
# torch.manual_seed(233)
# random.seed(233)
# torch.backends.cudnn.enabled = False

def train(train_iter, dev_iter, test_iter, model, text_field, label_field, args):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    clip = 3
    steps = 0
    best_dev = -1
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label.data.sub_(1)

            target =autograd.Variable(target)
            if args.use_cuda:
                feature, target = feature, target.cuda()

            optimizer.zero_grad()
            model.zero_grad()

            if feature.size(1) != args.batch_size:
                model.hidden = model.init_hidden(feature.size(1))
            else:
                model.hidden = model.init_hidden(args.batch_size)

            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()

            # clip gradients
            # utils.clip_grad_norm(model.parameters(), clip)  # clip=2 is the best 77
            # utils.clip_grad_norm(model.parameters(), max_norm=1e-4) 70
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_accuracy = eval(dev_iter, model, args)
                if dev_accuracy >= best_dev:
                    best_dev = dev_accuracy
                    test_eval(test_iter, model, args)
                    print('(epoch: %d, best dev acc = %.4f)\n' % (epoch, best_dev))
                else:
                    print('(epoch: %d, dev acc = %.4f, best dev acc = %.4f)\n' % (epoch, dev_accuracy, best_dev))

            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     # print("model", model)
            #     torch.save(model, save_path)
            #     test_eval(test_iter, model, save_path, args)
            #     model_count += 1
            #     # print("model_count \n", model_count)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label.data.sub_(1)

        target = autograd.Variable(target)
        if args.use_cuda:
            feature, target = feature, target.cuda()

        if feature.size(1) != args.batch_size:
            model.hidden = model.init_hidden(feature.size(1))
        else:
            model.hidden = model.init_hidden(args.batch_size)

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def test_eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label.data.sub_(1)
        # feature.data.t_()
        # target.data.sub_(1)  # batch first, index align
        target = autograd.Variable(target)
        if args.use_cuda:
            feature, target = feature, target.cuda()

        if feature.size(1) != args.batch_size:
            model.hidden = model.init_hidden(feature.size(1))
        else:
            model.hidden = model.init_hidden(args.batch_size)
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nTest Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    # file.write("model " + save_path + "\n")
    file.write("Test Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    file.write("\n")
    file.close()
