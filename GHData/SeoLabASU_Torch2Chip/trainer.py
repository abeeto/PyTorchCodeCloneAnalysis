"""
DNN Trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import accuracy, AverageMeter, print_table, lr_schedule, convert_secs2time, save_checkpoint

class BaseTrainer(object):
    def __init__(self,
        model: nn.Module,
        loss_type: str, 
        trainloader, 
        validloader,
        args,
        logger,
    ):  
        # model architecture
        self.model = model

        # args
        self.args = args

        # loader
        self.trainloader = trainloader
        self.validloader = validloader
        
        # loss func
        if loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type")
        
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=self.args.weight_decay)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        
        # learning rate scheduler
        if args.lr_sch == "step":
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[lr_schedule])
        elif args.lr_sch == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-5)

        
        if args.use_cuda:
            self.model = model.cuda()
            if args.ngpu > 1:
                self.model = nn.DataParallel(model)
                print("Data parallel!")

        # logger
        self.logger = logger
        self.logger_dict = {}

        # wandb logger
        if args.wandb:
            self.wandb_logger = wandb.init(entity=args.entity, project=args.project, name=args.name, config={"lr":args.lr})
            wandb.watch(self.model, log='all')
            self.wandb_logger.config.update(args)


    def base_forward(self, inputs, target):
        """Foward pass of NN
        """
        out = self.model(inputs)
        loss = self.criterion(out, target)
        return out, loss

    def base_backward(self, loss):
        # zero grad
        self.optimizer.zero_grad()
        loss.backward()
        for n, p in self.model.named_parameters():
            if "act_alpha" in n:
                self.alpha_grad.update(p.grad.item())

        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

        self.optimizer.step()
    
    def train_step(self, inputs, target):
        """Training step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        out, loss = self.base_forward(inputs, target)
        self.base_backward(loss)
        
        return out, loss

    def valid_step(self, inputs, target):
        """validation step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        out, loss = self.base_forward(inputs, target)
            
        return out, loss

    def train_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        beta = AverageMeter()

        self.model.train()

        for idx, (inputs, target) in enumerate(self.trainloader):
            if self.args.use_cuda:
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
            
            out, loss = self.train_step(inputs, target)
            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            if (idx+1) % 50 == 0:
                print("Train: [{}]/[{}], loss = {:.2f}; top1={:.2f}".format(idx+1, len(self.trainloader), loss.item(), prec1.item()))
        
        for name, param in self.model.named_parameters():
            if 'beta' in name:
                beta.update(param.item())

        self.logger_dict["train_loss"] = losses.avg
        self.logger_dict["train_top1"] = top1.avg
        self.logger_dict["train_top5"] = top5.avg


    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(self.validloader):
                if self.args.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)

                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        self.logger.info("\nStart training: lr={}, loss={}, optim={}".format(self.args.lr, self.args.loss_type, self.args.optimizer))

        start_time = time.time()
        epoch_time = AverageMeter()
        best_acc = 0.
        for epoch in range(self.args.epochs):
            self.logger_dict["ep"] = epoch+1
            self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
            
            # training and validation
            self.train_epoch()
            self.valid_epoch()
            self.lr_scheduler.step()

            is_best = self.logger_dict["valid_top1"] > best_acc
            if is_best:
                best_acc = self.logger_dict["valid_top1"]

            state = {
                'state_dict': self.model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
            }

            filename=f"checkpoint.pth.tar"
            save_checkpoint(state, is_best, self.args.save_path, filename=filename)

            # online log
            if self.args.wandb:
                self.wandb_logger.log(self.logger_dict)

            # terminal log
            columns = list(self.logger_dict.keys())
            values = list(self.logger_dict.values())
            print_table(values, columns, epoch, self.logger)

            # record time
            e_time = time.time() - start_time
            epoch_time.update(e_time)
            start_time = time.time()

            need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (self.args.epochs - epoch))
            print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs))
