import argparse

import torch
import tqdm

from lib.build.registry import Registries
from lib.models.backbones.resnet import *
from lib.models.loss import FocalLoss
from lib.datasets.cifar import *
from lib.utils import transforms
from lib.utils.evaluator import Evaluator
from lib.utils.logger import Logger
from lib.utils.lr_scheduler import WarmUpStepLR
from lib.utils.saver import Saver


class Trainer:
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)

        # Define Logger
        self.logger = Logger(args.save_path)

        # Define Evaluator
        self.evaluator = Evaluator(args.num_classes)

        # Define Best Prediction
        self.best_pred = 0.0

        # Define Last Epoch
        self.last_epoch = -1

        # Define DataLoader
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        target_transform = transforms.Compose([
            transforms.ToLong(),
        ])
        train_dataset = Registries.dataset_registry.__getitem__(args.dataset)(args.dataset_path, 'train',
                                                                              train_transform, target_transform)
        valid_dataset = Registries.dataset_registry.__getitem__(args.dataset)(args.dataset_path, 'valid',
                                                                              valid_transform, target_transform)

        kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True}

        self.train_loader = DataLoader(dataset=train_dataset,
                                       shuffle=False,
                                       **kwargs)
        self.valid_loader = DataLoader(dataset=valid_dataset,
                                       shuffle=False,
                                       **kwargs)

        # Define Model
        self.model = Registries.backbone_registry.__getitem__(args.backbone)(num_classes=10)

        # Define Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.init_learning_rate, momentum=0.9,
                                         dampening=0.1)

        # Define Criterion
        self.criterion = FocalLoss()

        # Define  Learning Rate Scheduler
        self.scheduler = WarmUpStepLR(self.optimizer, warm_up_end_epoch=100, step_size=50, gamma=0.1)

        # Use cuda
        if torch.cuda.is_available() and args.use_gpu:
            self.device = torch.device("cuda", args.gpu_ids[0])
            if len(args.gpu_ids) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # Use pretrained model
        if args.pretrained_model_path is not None:
            if not os.path.isfile(args.pretrained_model_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrained_model_path))
            else:
                checkpoint = torch.load(args.pretrained_model_path)
                if args.use_gpu and len(args.gpu_ids) > 1:
                    self.model.module.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint['model'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.best_pred = checkpoint['best_pred']
                self.optimizer = self.scheduler.optimizer
                self.last_epoch = checkpoint['last_epoch']
                print("=> loaded checkpoint '{}'".format(args.pretrained_model_path))

    def train(self):
        for epoch in range(self.last_epoch + 1, self.args.num_epochs):
            self._train_a_epoch(epoch)
            if epoch % self.args.valid_step == (self.args.valid_step - 1):
                self._valid_a_epoch(epoch)

    def _train_a_epoch(self, epoch):
        print('train epoch %d' % epoch)
        total_loss = 0
        tbar = tqdm.tqdm(self.train_loader)
        self.model.train()  # change the model to train mode
        step_num = len(self.train_loader)
        for step, sample in enumerate(tbar):
            inputs, labels = sample['data'], sample['label']  # get the inputs and labels from dataloader
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if epoch == 0 and step == 0:
                self.logger.show_img_grid(inputs)
                self.logger.writer.add_graph(self.model, inputs)
            self.optimizer.zero_grad()  # zero the optimizer because the gradient will accumulate in PyTorch
            outputs = self.model(inputs)  # get the output(forward)
            loss = self.criterion(outputs, labels)  # compute the loss
            loss.backward()  # back propagate the loss(backward)
            total_loss += loss.item()
            self.optimizer.step()  # update the weights
            tbar.set_description('train iteration loss= %.6f' % loss.item())
            self.logger.writer.add_scalar('train iteration loss', loss, epoch * step_num + step)
        self.logger.writer.add_scalar('train epoch loss', total_loss / step_num, epoch)
        preds = torch.argmax(outputs, dim=1)
        self.logger.add_pr_curve_tensorboard('pr curve', labels, preds)
        self.scheduler.step()  # update the learning rate
        self.saver.save_checkpoint({'scheduler': self.scheduler.state_dict(),
                                    'model': self.model.state_dict(),
                                    'best_pred': self.best_pred,
                                    'last_epoch': epoch},
                                   'current_checkpoint.pth')

    def _valid_a_epoch(self, epoch):
        print('valid epoch %d' % epoch)
        tbar = tqdm.tqdm(self.valid_loader)
        self.model.eval()  # change the model to eval mode
        with torch.no_grad():
            for step, sample in enumerate(tbar):
                inputs, labels = sample['data'], sample['label']  # get the inputs and labels from dataloader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)  # get the output(forward)
                predicts = torch.argmax(outputs, dim=1)
                self.evaluator.add_batch(labels.cpu().numpy(), predicts.cpu().numpy())
        new_pred = self.evaluator.Mean_Intersection_over_Union()
        print()

        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.saver.save_checkpoint({'scheduler': self.scheduler.state_dict(),
                                        'model': self.model.state_dict(),
                                        'best_pred': self.best_pred,
                                        'last_epoch': epoch},
                                       'best_checkpoint.pth')
            self.saver.save_parameters()


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--valid_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset you are using.')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone you are using.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--init_learning_rate', type=float, default=0.001, help='init learning rate used for train')
    parser.add_argument('--dataset_path', type=str, default='./data/cifar-10-batches-py/', help='path to dataset')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=10, help='num of object classes (with void)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to load pretrained model')
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help='path to save pretrained model and results')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    if args.use_gpu:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # start to train
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
