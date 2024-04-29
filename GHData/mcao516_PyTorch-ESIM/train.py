#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import classification_report

from ESIM import ESIM
from utils import Progbar, to_device


class Model:
    """Enhanced Sequential Inference Model (ESIM) for natural language inference.
    """
    def __init__(self, args):
        """Class initialization.
        """
        self.args = args
        self.logger = args.logger

        # initialziation
        self.model = self._build_model()
        self.model.to(args.device)
        self._initialize_model(self.model)
        self.optimizer = self._get_optimizer(self.model)
        self.scheduler = self._get_scheduler(self.optimizer)
        self.criterion = self._get_criterion()

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.logger.info("- Let's use {} GPUs !".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.info("- Train the model on single GPU :/")

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            self.logger.info("- Let's do distributed training !")
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        # tensorboard
        if args.write_summary and args.local_rank in [-1, 0]:
            self.logger.info("- Let's use tensorboard on local rank {} device :)".format(args.local_rank))
            self.writer = SummaryWriter(self.args.summary_path)

    def _build_model(self):
        """Build ESIM model.
        """
        return ESIM(self.args.vector_size,
                    self.args.vocab_size,
                    self.args.hidden_size,
                    self.args.class_num,
                    self.args.dropout)

    def _initialize_model(self, model):
        """Initialize ESIM model paramerters.
        """
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, a=-0.1, b=0.1)

    def initialize_embeddings(self, vectors):
        """Load pre-trained word embeddings.
        """
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_embeddings(vectors)
        else:
            self.model.load_embeddings(vectors)

    def _set_parameter_requires_grad(self):
        """Specify which parameters need compute gradients.
        """
        # we don't need this right now
        pass

    def _get_optimizer(self, model):
        """Get optimizer for model training.
        """
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            self.logger.info("Unknow optimizer: {}, exiting...".format(self.args.optimizer))
            exit()

        return optimizer

    def _get_scheduler(self, optimizer):
        """Get scheduler for adjusting learning rate.
        """
        return MultiStepLR(optimizer, milestones=[25], gamma=0.1)

    def _get_criterion(self):
        """Loss function.
        """
        return nn.CrossEntropyLoss()

    def load_weights(self, model_dir):
        """Load pre-trained model weights.
        """
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "esim.pickle")))

    def save_model(self, model_dir=None):
        """Save model's weights.
        """
        if not model_dir:
            model_dir = self.args.model_dir
        torch.save(self.model.state_dict(), os.path.join(model_dir, "esim.pickle"))
        self.logger.info("- ESIM model is saved at: {}".format(
            os.path.join(model_dir, "esim.pickle")))

    def loss_batch(self, p, h, labels, criterion, optimizer=None):
        """
        Arguments:
            p {torch.Tensor} -- premise [batch, seq_len]
            h {torch.Tensor} -- hypothesis [batch, seq_len]
            labels {torch.Tensor} -- hypothesis [batch]
            criterion {torch.nn.Loss} -- loss function

        Keyword Arguments:
            optimizer {torch.optim.Optimizer} -- PyTorch optimizer

        Returns:
            logits {torch.Tensor} -- raw, unnormalized scores for each class
                with shape [batch, class_num]
        """
        logits = self.model(p, h)
        loss = criterion(logits, labels)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if optimizer is not None:
            with torch.set_grad_enabled(True):
                loss.backward()  # compute gradients
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.max_grad_norm)
                optimizer.step()  # update model parameters
                optimizer.zero_grad()  # clean all gradients

        return loss.item(), logits.detach()

    def train_epoch(self, train_iter, criterion, optimizer, epoch):
        """Train the model for one single epoch.
        """
        self.model.train()  # set the model to training mode
        prog = Progbar(target=len(train_iter))

        train_loss = 0.0
        for i, batch in enumerate(train_iter):
            batch_data = to_device(batch, self.args.device)
            batch_loss, _ = self.loss_batch(batch_data['premise'],
                                            batch_data['hypothesis'],
                                            batch_data['label'],
                                            criterion,
                                            optimizer=optimizer)
            train_loss += batch_loss
            prog.update(i + 1, [("train loss", batch_loss)])

            if self.args.local_rank in [-1, 0] and self.writer:
                self.writer.add_scalar('batch_loss', batch_loss, epoch*len(train_iter) + i + 1)

        # compute the average loss (batch loss)
        epoch_loss = train_loss / len(train_iter)

        # update scheduler
        self.scheduler.step()

        return epoch_loss

    def evaluate(self, dev_iter, criterion):
        """Evaluate the model.
        """
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            eval_loss, eval_corrects = 0.0, 0.0
            for _, batch in enumerate(dev_iter):
                batch_data = to_device(batch, self.args.device)
                batch_loss, outputs = self.loss_batch(batch_data['premise'],
                                                      batch_data['hypothesis'],
                                                      batch_data['label'],
                                                      criterion,
                                                      optimizer=None)
                _, preds = torch.max(outputs, 1)  # preds: [batch_size]

                eval_loss += batch_loss
                eval_corrects += torch.sum(preds == (batch_data['label'])).double()

            avg_loss = eval_loss / len(dev_iter)
            avg_acc = eval_corrects / len(dev_iter.dataset)

        return avg_loss, avg_acc

    def fit(self, train_iter, dev_iter):
        """Model training and evaluation.
        """
        best_acc = 0.
        num_epochs = self.args.num_epochs

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # training
            train_loss = self.train_epoch(train_iter, self.criterion, self.optimizer, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # evaluation, only on the master node
            if self.args.local_rank in [-1, 0]:
                eval_loss, eval_acc = self.evaluate(dev_iter, self.criterion)
                self.logger.info("Evaluation:")
                self.logger.info("- loss: {}".format(eval_loss))
                self.logger.info("- acc: {}".format(eval_acc))

                # monitor loss and accuracy
                if self.writer:
                    self.writer.add_scalar('epoch_loss', train_loss, epoch)
                    self.writer.add_scalar('eval_loss', eval_loss, epoch)
                    self.writer.add_scalar('eval_acc', eval_acc, epoch)
                    # self.writer.add_scalar('lr', self.scheduler.get_lr()[0])

                # save the model
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.logger.info("New best score!")
                    self.save_model()

    def predict(self, premise, hypothesis):
        """Prediction.

        Arguments:
            premise {torch.Tensor} -- [batch, seq_len]
            hypothesis {torch.Tensor} -- [batch, seq_len]

        Returns:
            pres {torch.Tensor} -- [batch]
        """
        self.model.eval()  # evaluation mode
        with torch.no_grad():
            outputs = self.model(premise, hypothesis)  # outpus: [batch, num_classes]
            _, preds = torch.max(outputs, 1)
        return preds

    def get_report(self, dataset, target_names=None):
        """Test the model and print out a report.
        """
        pred_class, label_class = [], []
        for batch in dataset:
            batch_data = to_device(batch, self.args.device)
            preds = self.predict(batch_data['premise'], batch_data['hypothesis'])
            pred_class += preds.tolist()
            label_class += (batch_data['label']).tolist()

        self.logger.info('\n')
        self.logger.info(classification_report(label_class, pred_class,
                                               target_names=target_names))
        return pred_class, label_class