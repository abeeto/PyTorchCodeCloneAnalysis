import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import random
torch.manual_seed(233)
random.seed(233)
import torch.optim.lr_scheduler as lr_scheduler

def train(train_iter, dev_iter, test_iter, model, text_field, label_field, args):
    if args.cuda:
        model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4, weight_decay=1e-8)       # eps ?
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)
    # modify learning rate
    # method 1
    # args.lr = args.lr * (1.0 - 1e-5 * epoch)
    # method 2
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.99)

    clip = 3
    steps = 0
    model.train()
    # total = args.epochs * len(train_iter)
    # print(total)
    # print(len(train_iter))
    # print(len(dev_iter))

    for epoch in range(1, args.epochs+1):
        # scheduler.step()
        for batch in train_iter:
            feature, target = batch.text, batch.label           # feature [torch.LongTensor of size 16x36], target Variable [torch.LongTensor of size 16]
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            # scheduler.step(loss)
            # clip gradients
            # utils.clip_grad_norm(model.parameters(), clip)      # 经验值

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,train_size,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                # eval(dev_iter, model, args, scheduler)
                eval(dev_iter, model, args)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
                test_eval(test_iter, model, save_path, args)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        # feature, target = batch.text, batch.label
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # if args.cuda:
        #     feature, target = feature.cuda(), feature.cuda()
        #
        # logit = model(feature)
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        # scheduler.step(loss.data[0])

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

def test_eval(data_iter, model, save_path, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

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
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss,
                                                                          accuracy,
                                                                          corrects,
                                                                          size))
    file.write("\n")
    file.close()

def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    # print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]

