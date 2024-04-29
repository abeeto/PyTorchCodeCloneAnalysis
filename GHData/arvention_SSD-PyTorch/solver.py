import os
import os.path as osp
import numpy as np
import torch
import time
import datetime
import pickle
import torch.optim as optim
from tqdm import tqdm
from utils.genutils import to_var
from utils.nms_wrapper import nms

from models.model import get_model
from loss.loss import get_loss
from layers.anchor_box import AnchorBox
from utils.timer import Timer

from data.pascal_voc import save_results as voc_save, do_python_eval


class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, train_loader, test_loader, config):
        """
        Initializes a Solver object
        """

        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.build_model()

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()
        else:
            self.model.init_weights(self.model_save_path, self.basenet)

    def build_model(self):
        """
        Instantiate the model, loss criterion, and optimizer
        """

        # instantiate anchor boxes
        anchor_boxes = AnchorBox(self.new_size, self.anchor_config)
        self.anchor_boxes = anchor_boxes.get_boxes()
        if torch.cuda.is_available() and self.use_gpu:
            self.anchor_boxes = self.anchor_boxes.cuda()

        # instatiate model
        self.model = get_model(config=self.config,
                               anchors=self.anchor_boxes)

        # instatiate loss criterion
        self.criterion = get_loss(config=self.config)

        # instatiate optimizer
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        # print network
        self.print_network(self.model)

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def print_network(self, model):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        print('loaded trained model ver {}'.format(self.pretrained_model))

    def adjust_learning_rate(self,
                             optimizer,
                             gamma,
                             step,
                             i=None,
                             iters_per_epoch=None,
                             epoch=None):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if self.warmup and epoch < self.warmup_step:
            lr = 1e-6 + (self.lr-1e-6) * i / (iters_per_epoch * 5)
        else:
            lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_loss_log(self,
                       start_time,
                       cur,
                       total,
                       class_loss,
                       loc_loss,
                       loss):
        """
        Prints the loss and elapsed time for each epoch
        """

        elapsed = time.time() - start_time
        total_time = (total - cur) * elapsed / (cur + 1)

        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {} -- {}, {} [{}/{}]\n" \
              "class_loss: {:.4f}, loc_loss: {:.4f}, " \
              "loss: {:.4f}".format(elapsed,
                                    total_time,
                                    self.counter,
                                    cur + 1,
                                    total,
                                    class_loss.item(),
                                    loc_loss.item(),
                                    loss.item())

        print(log)

    def save_model(self, i):
        """
        Saves a model per i iteration
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, i + 1)
        )

        torch.save(self.model.state_dict(), path)

    def model_step(self, images, targets, count):
        """
        A step for each iteration
        """

        if count == 0:
            # update parameters
            self.optimizer.step()

            # empty the gradients of the model through the optimizer
            self.optimizer.zero_grad()

            count = self.batch_multiplier

        # forward pass
        class_preds, loc_preds = self.model(images)

        # compute loss
        class_targets = [target[:, -1] for target in targets]
        loc_targets = [target[:, :-1] for target in targets]
        losses = self.criterion(class_preds=class_preds,
                                class_targets=class_targets,
                                loc_preds=loc_preds,
                                loc_targets=loc_targets,
                                anchors=self.anchor_boxes)
        class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        loss = loss / self.batch_multiplier
        loss.backward()

        count = count - 1

        # return loss
        return class_loss, loc_loss, loss, count

    def train_iter(self, start):
        step_index = 0
        start_time = time.time()
        batch_iterator = iter(self.train_loader)
        count = 0

        for i in range(start, self.num_iterations):

            if i in self.sched_milestones:
                step_index += 1
                self.adjust_learning_rate(optimizer=self.optimizer,
                                          gamma=self.sched_gamma,
                                          step=step_index)

            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self.train_loader)
                images, targets = next(batch_iterator)

            images = to_var(images, self.use_gpu)
            targets = [to_var(target, self.use_gpu) for target in targets]

            class_loss, loc_loss, loss, count = self.model_step(images,
                                                                targets,
                                                                count)

            # print out loss log
            if (i + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    cur=i,
                                    total=self.num_iterations,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)
                self.losses.append([i, class_loss, loc_loss, loss])

            # save model
            if (i + 1) % self.model_save_step == 0:
                self.save_model(i)

        self.save_model(i)

    def train_epoch(self, start):
        step_index = 0
        start_time = time.time()
        iters_per_epoch = len(self.train_loader)

        for e in range(start, self.num_epochs):

            if e in self.sched_milestones:
                step_index += 1

            for i, (images, targets) in enumerate(tqdm(self.train_loader)):
                self.adjust_learning_rate(optimizer=self.optimizer,
                                          gamma=self.sched_gamma,
                                          step=step_index,
                                          i=i,
                                          iters_per_epoch=iters_per_epoch,
                                          epoch=e)

                images = to_var(images, self.use_gpu)
                targets = [to_var(target, self.use_gpu) for target in targets]

                class_loss, loc_loss, loss = self.model_step(images, targets)

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    cur=e,
                                    total=self.num_epochs,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)
                self.losses.append([e, class_loss, loc_loss, loss])

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

        self.save_model(e)

    def train(self):
        """
        training process
        """

        # set model in training mode
        self.model.train()

        self.losses = []

        # start with a trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        if self.counter == 'iter':
            self.train_iter(start)
        elif self.counter == 'epoch':
            self.train_epoch(start)

        # print losses
        print('\n--Losses--')
        for i, class_loss, loc_loss, loss in self.losses:
            print(i, "{:.4f} {:.4f} {:.4f}".format(class_loss.item(),
                                                   loc_loss.item(),
                                                   loss.item()))

    def eval(self, dataset, max_per_image, threshold):
        num_images = len(dataset)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.class_count)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        results_path = osp.join(self.result_save_path,
                                self.pretrained_model)
        det_file = os.path.join(results_path, 'detections.pkl')

        detect_times = []
        nms_times = []

        with torch.no_grad():
            for i in range(num_images):
                image, target, h, w = dataset.pull_item(i)
                image = to_var(image.unsqueeze(0), self.use_gpu)

                _t['im_detect'].tic()
                boxes, scores = self.model(image)
                detect_time = _t['im_detect'].toc(average=False)
                detect_times.append(detect_time)
                boxes = boxes[0]
                scores = scores[0]

                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                # scale each detection back up to the image
                scale = torch.Tensor([w, h, w, h]).cpu().numpy()
                boxes *= scale

                _t['misc'].tic()

                for j in range(1, self.class_count):
                    inds = np.where(scores[:, j] > threshold)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes[inds]
                    c_scores = scores[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)

                    keep = nms(c_dets, 0.45, force_cpu=True)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self.class_count)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, self.class_count):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                nms_time = _t['misc'].toc(average=False)
                nms_times.append(nms_time)

                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                      .format(i + 1, num_images, detect_time, nms_time))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        if self.dataset == 'voc':
            voc_save(all_boxes, dataset, results_path)
            do_python_eval(results_path, dataset)

        detect_times = np.asarray(detect_times)
        nms_times = np.asarray(nms_times)
        total_times = np.add(detect_times, nms_times)

        print('fps[all]:', (1 / np.mean(detect_times[1:])))
        print('fps[all]:', (1 / np.mean(nms_times[1:])))
        print('fps[all]:', (1 / np.mean(total_times[1:])))

    def test(self):
        """
        testing process
        """
        self.model.eval()
        self.eval(dataset=self.test_loader.dataset,
                  max_per_image=300,
                  threshold=0.005)
