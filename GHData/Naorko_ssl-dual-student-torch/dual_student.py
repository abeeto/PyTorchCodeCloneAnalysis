import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable

from src import architectures, ramps, mt_func, losses
from src.data import NO_LABEL
from src.mt_func import accuracy
from src.utils import *

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def create_model(name, num_classes, ema=False):
    LOG.info('=> creating {name} model: {arch}'.format(
        name=name,
        arch=args.arch))

    model_factory = architectures.__dict__[args.arch]
    model_params = dict(num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # decline lr
    lr *= ramps.zero_cosine_rampdown(epoch, args.epochs)

    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr


def validate(eval_loader, model, log):
    global global_step
    data_size = len(eval_loader.dataset)
    meters = AverageMeterSet()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        y_prob, y_true = [], []
        for i, (inp, target) in enumerate(eval_loader):
            labeled_minibatch_size = target.data.ne(NO_LABEL).sum().item()
            assert labeled_minibatch_size > 0

            # compute output and update inference time
            inf_start = time.time()
            output1, _ = model(inp)
            inf_time = time.time() - inf_start
            meters.update('inference_time', inf_time, n=labeled_minibatch_size)

            softmax1 = F.softmax(output1, dim=1).cpu()

            y_prob.append(softmax1)
            y_true.append(target)

        y_prob, y_true = [torch.cat(ys, dim=0) for ys in [y_prob, y_true]]
        y_pred = torch.argmax(y_prob, dim=1)

        # update TPR, FPR and precision (using confusion matrix)
        cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP, FN, TP, TN = FP.sum(), FN.sum(), TP.sum(), TN.sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        Precision = TP / (TP + FP)
        meters.update('TPR', TPR, n=data_size)
        meters.update('FPR', FPR, n=data_size)
        meters.update('Precision', Precision, n=data_size)

        auc = metrics.roc_auc_score(y_true, y_prob, multi_class='ovr')
        meters.update('AUC', auc, n=data_size)

        num_classes = len(eval_loader.dataset.dataset.classes)
        y_true_one_hot = label_binarize(y_true, classes=[*range(num_classes)])
        pr_curves = [metrics.average_precision_score(y_true_one_hot[:, i], y_prob[:, i]) for i in range(num_classes)]
        pr_curve = np.mean(pr_curves)
        meters.update('PR-Curve', pr_curve, n=data_size)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(y_prob, y_true, topk=(1, 5))
        prec1, prec5 = prec1.item(), prec5.item()

        meters.update('Accuracy-top1', prec1, data_size)
        meters.update('Accuracy-top5', prec5, data_size)

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
             .format(top1=meters['Accuracy-top1'], top5=meters['Accuracy-top5']))

    log.record(global_step, {
        **meters.values()
    })

    global_step += 1

    return meters


def train_epoch(train_loader, l_model, r_model, l_optimizer, r_optimizer, epoch, log):
    global global_step

    meters = AverageMeterSet()

    # define criterions
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    residual_logit_criterion = losses.symmetric_mse_loss

    consistency_criterion = losses.softmax_mse_loss
    stabilization_criterion = losses.softmax_mse_loss

    l_model.train()
    r_model.train()

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        # adjust learning rate
        adjust_learning_rate(l_optimizer, epoch, i, len(train_loader))
        adjust_learning_rate(r_optimizer, epoch, i, len(train_loader))
        meters.update('l_lr', l_optimizer.param_groups[0]['lr'])
        meters.update('r_lr', r_optimizer.param_groups[0]['lr'])

        # prepare data
        l_input_var = Variable(l_input)
        r_input_var = Variable(r_input)
        le_input_var = Variable(r_input, requires_grad=False, volatile=True)
        re_input_var = Variable(l_input, requires_grad=False, volatile=True)
        target_var = Variable(target.cuda(non_blocking=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size >= 0 and unlabeled_minibatch_size >= 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        meters.update('unlabeled_minibatch_size', unlabeled_minibatch_size)

        # forward
        l_model_out = l_model(l_input_var)
        r_model_out = r_model(r_input_var)
        le_model_out = l_model(le_input_var)
        re_model_out = r_model(re_input_var)

        if isinstance(l_model_out, Variable):
            assert args.logit_distance_cost < 0
            l_logit1 = l_model_out
            r_logit1 = r_model_out
            le_logit1 = le_model_out
            re_logit1 = re_model_out
        elif len(l_model_out) == 2:
            assert len(r_model_out) == 2
            l_logit1, l_logit2 = l_model_out
            r_logit1, r_logit2 = r_model_out
            le_logit1, le_logit2 = le_model_out
            re_logit1, re_logit2 = re_model_out

        # logit distance loss from mean teacher
        if args.logit_distance_cost >= 0:
            l_class_logit, l_cons_logit = l_logit1, l_logit2
            r_class_logit, r_cons_logit = r_logit1, r_logit2
            le_class_logit, le_cons_logit = le_logit1, le_logit2
            re_class_logit, re_cons_logit = re_logit1, re_logit2

            l_res_loss = args.logit_distance_cost * residual_logit_criterion(l_class_logit,
                                                                             l_cons_logit) / minibatch_size
            r_res_loss = args.logit_distance_cost * residual_logit_criterion(r_class_logit,
                                                                             r_cons_logit) / minibatch_size
            meters.update('l_res_loss', l_res_loss.item())
            meters.update('r_res_loss', r_res_loss.item())
        else:
            l_class_logit, l_cons_logit = l_logit1, l_logit1
            r_class_logit, r_cons_logit = r_logit1, r_logit1
            le_class_logit, le_cons_logit = le_logit1, le_logit1
            re_class_logit, re_cons_logit = re_logit1, re_logit1

            l_res_loss = 0.0
            r_res_loss = 0.0
            meters.update('l_res_loss', 0.0)
            meters.update('r_res_loss', 0.0)

        # classification loss
        l_class_loss = class_criterion(l_class_logit, target_var) / minibatch_size
        r_class_loss = class_criterion(r_class_logit, target_var) / minibatch_size
        meters.update('l_class_loss', l_class_loss.item())
        meters.update('r_class_loss', r_class_loss.item())

        l_loss, r_loss = l_class_loss, r_class_loss
        l_loss += l_res_loss
        r_loss += r_res_loss

        # consistency loss
        consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

        le_class_logit = Variable(le_class_logit.detach().data, requires_grad=False)
        l_consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, le_class_logit) / minibatch_size
        meters.update('l_cons_loss', l_consistency_loss.item())
        l_loss += l_consistency_loss

        re_class_logit = Variable(re_class_logit.detach().data, requires_grad=False)
        r_consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, re_class_logit) / minibatch_size
        meters.update('r_cons_loss', r_consistency_loss.item())
        r_loss += r_consistency_loss

        # stabilization loss
        # value (cls_v) and index (cls_i) of the max probability in the prediction
        l_cls_v, l_cls_i = torch.max(F.softmax(l_class_logit, dim=1), dim=1)
        r_cls_v, r_cls_i = torch.max(F.softmax(r_class_logit, dim=1), dim=1)
        le_cls_v, le_cls_i = torch.max(F.softmax(le_class_logit, dim=1), dim=1)
        re_cls_v, re_cls_i = torch.max(F.softmax(re_class_logit, dim=1), dim=1)

        l_cls_i = l_cls_i.data.cpu().numpy()
        r_cls_i = r_cls_i.data.cpu().numpy()
        le_cls_i = le_cls_i.data.cpu().numpy()
        re_cls_i = re_cls_i.data.cpu().numpy()

        # stable prediction mask 
        l_mask = (l_cls_v > args.stable_threshold).data.cpu().numpy()
        r_mask = (r_cls_v > args.stable_threshold).data.cpu().numpy()
        le_mask = (le_cls_v > args.stable_threshold).data.cpu().numpy()
        re_mask = (re_cls_v > args.stable_threshold).data.cpu().numpy()

        # detach logit -> for generating stablilization target 
        in_r_cons_logit = Variable(r_cons_logit.detach().data, requires_grad=False)
        tar_l_class_logit = Variable(l_class_logit.clone().detach().data, requires_grad=False)

        in_l_cons_logit = Variable(l_cons_logit.detach().data, requires_grad=False)
        tar_r_class_logit = Variable(r_class_logit.clone().detach().data, requires_grad=False)

        # generate target for each sample
        for sdx in range(0, minibatch_size):

            # Check if stable
            l_stable = False
            if l_mask[sdx] == 0 and le_mask[sdx] == 0:
                # unstable: do not satisfy the 2nd condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
            elif l_cls_i[sdx] != le_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
            else:
                l_stable = True

            r_stable = False
            if r_mask[sdx] == 0 and re_mask[sdx] == 0:
                # unstable: do not satisfy the 2nd condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
            elif r_cls_i[sdx] != re_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
            else:
                r_stable = True

            # calculate stability if both models are stable for a sample
            if l_stable and r_stable:
                # compare by consistency
                l_sample_cons = consistency_criterion(l_cons_logit[sdx:sdx + 1, ...],
                                                      le_class_logit[sdx:sdx + 1, ...]).item()
                r_sample_cons = consistency_criterion(r_cons_logit[sdx:sdx + 1, ...],
                                                      re_class_logit[sdx:sdx + 1, ...]).item()
                if l_sample_cons < r_sample_cons:
                    # loss: l -> r
                    tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
                elif l_sample_cons > r_sample_cons:
                    # loss: r -> l
                    tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]

        # calculate stablization weight
        stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch, args.stabilization_rampup)
        stabilization_weight = (unlabeled_minibatch_size / minibatch_size) * stabilization_weight

        # stabilization loss for r model
        for idx in range(unlabeled_minibatch_size, minibatch_size):
            tar_l_class_logit[idx, ...] = in_r_cons_logit[idx, ...]

        r_stabilization_loss = stabilization_weight * stabilization_criterion(r_cons_logit,
                                                                              tar_l_class_logit) / unlabeled_minibatch_size
        meters.update('r_stable_loss', r_stabilization_loss.item())
        r_loss += r_stabilization_loss

        # stabilization loss for l model
        for idx in range(unlabeled_minibatch_size, minibatch_size):
            tar_r_class_logit[idx, ...] = in_l_cons_logit[idx, ...]

        l_stabilization_loss = stabilization_weight * stabilization_criterion(l_cons_logit,
                                                                              tar_r_class_logit) / unlabeled_minibatch_size

        meters.update('l_stable_loss', l_stabilization_loss.item())
        l_loss += l_stabilization_loss

        if np.isnan(l_loss.item()) or np.isnan(r_loss.item()):
            LOG.info('Loss value equals to NAN!')
            continue
        assert not (l_loss.item() > 1e5), 'L-Loss explosion: {}'.format(l_loss.item())
        assert not (r_loss.item() > 1e5), 'R-Loss explosion: {}'.format(r_loss.item())
        meters.update('l_loss', l_loss.item())
        meters.update('r_loss', r_loss.item())

        # calculate prec and error
        l_prec = mt_func.accuracy(l_class_logit.data, target_var.data, topk=(1,))[0].item()
        r_prec = mt_func.accuracy(r_class_logit.data, target_var.data, topk=(1,))[0].item()

        meters.update('l_top1', l_prec, labeled_minibatch_size)
        meters.update('l_error1', 100. - l_prec, labeled_minibatch_size)

        meters.update('r_top1', r_prec, labeled_minibatch_size)
        meters.update('r_error1', 100. - r_prec, labeled_minibatch_size)

        # update model
        l_optimizer.zero_grad()
        l_loss.backward()
        l_optimizer.step()

        r_optimizer.zero_grad()
        r_loss.backward()
        r_optimizer.step()

        # record
        global_step += 1
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Batch-T {meters[batch_time]:.3f}\t'
                     'L-Class {meters[l_class_loss]:.4f}\t'
                     'R-Class {meters[r_class_loss]:.4f}\t'
                     'L-Res {meters[l_res_loss]:.4f}\t'
                     'R-Res {meters[r_res_loss]:.4f}\t'
                     'L-Cons {meters[l_cons_loss]:.4f}\t'
                     'R-Cons {meters[r_cons_loss]:.4f}\n'
                     'L-Stable {meters[l_stable_loss]:.4f}\t'
                     'R-Stable {meters[r_stable_loss]:.4f}\t'
                     'L-Prec@1 {meters[l_top1]:.3f}\t'
                     'R-Prec@1 {meters[r_top1]:.3f}\t'
                     .format(epoch, i, len(train_loader), meters=meters))

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()})
