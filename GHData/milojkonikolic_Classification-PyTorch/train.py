import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

from utils.utils import get_device, get_logger, get_tb_writer, copy_config
from models.common import load_model, save_model
from utils.lr_schedulers import StepDecayLR
from utils.dataset import create_dataloaders


class Train():

    def __init__(self, config):
        self.logger = get_logger()
        self.writer = get_tb_writer(config["Logging"]["tb_logdir"])
        self.num_classes = config["Dataset"]["num_classes"]
        self.batch_size = config["Train"]["batch_size"]
        self.epochs = config["Train"]["epochs"]
        self.input_shape = config["Train"]["image_size"]
        self.channels = config["Train"]["channels"]
        self.ckpt_dir = config["Logging"]["ckpt_dir"]
        self.eval_per_epoch = config["Train"]["eval_per_epoch"]
        self.device = get_device(config["Train"]["device"])
        self.model = load_model(config["Train"]["arch"], self.num_classes, self.device,
                                config["Train"]["pretrained"], self.channels, self.logger)
        self.optimizer = self.get_optimizer(config["Train"]["optimizer"], config["Train"]["lr_scheduler"]["lr_init"])
        self.scheduler = self.get_scheduler(config["Train"]["lr_scheduler"])
        self.train_loader, self.val_loader, self.classes = \
            create_dataloaders(train_data_path=config["Dataset"]["train_data_path"],
                               val_data_path=config["Dataset"]["val_data_path"],
                               classes_path=config["Dataset"]["classes_path"],
                               img_size=self.input_shape,
                               batch_size=self.batch_size,
                               augment=config["Augmentation"],
                               logger=self.logger)
        self.criterion = nn.CrossEntropyLoss()

    def get_optimizer(self, opt, lr=0.001):
        """
        Args
            opt: string, optimizer from config file
            model: nn.Module, generated model
            lr: float, specified learning rate
        Returns:
            optimizer
        """

        if opt.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                      f"For supported optimizers see documentation")
        return optimizer

    def get_scheduler(self, lr_scheduler_dict):
        if lr_scheduler_dict["StepDecay"]["use"]:
            lr_scheduler = StepDecayLR(optimizer=self.optimizer,
                                       lr_init=lr_scheduler_dict["lr_init"],
                                       lr_end=lr_scheduler_dict["lr_end"],
                                       epoch_steps=lr_scheduler_dict["StepDecay"]["epoch_steps"])
            return lr_scheduler
        else:
            return False

    def train(self):

        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        total_num_params = 0
        for p in params:
            total_num_params += p
        self.logger.info(f"Number of parameters of the model: {total_num_params}")

        global_step = 0
        eval_steps = list(np.linspace(len(self.train_loader) // self.eval_per_epoch, len(self.train_loader),
                                      self.eval_per_epoch).astype(int))
        log_step = len(self.train_loader) // 20
        best_accuracy = 0

        self.logger.info(f"---------------- Training Started ----------------")
        for epoch in range(self.epochs):
            epoch += 1
            for batch, (X_train, y_train) in enumerate(self.train_loader):
                batch += 1

                X_train = X_train.cuda(self.device)
                y_train = y_train.cuda(self.device)
                y_pred = self.model(X_train)
                train_loss = self.criterion(y_pred, y_train)

                global_step += 1

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step(epoch)

                self.writer.add_scalar("train_loss", train_loss.item(), global_step)
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], global_step)

                if batch % log_step == 0:
                    self.logger.info(f"epoch: {epoch}/{self.epochs}, "
                                     f"batch: {batch}/{len(self.train_loader)}, "
                                     f"train_loss: {train_loss.item()}")

                if batch in eval_steps:
                    val_corr = 0
                    with torch.no_grad():
                        for val_batch, (X_val, y_val) in enumerate(self.val_loader):
                            val_batch += 1
                            X_val = X_val.cuda(self.device)
                            y_val = y_val.cuda(self.device)
                            val_pred = self.model(X_val)
                            predicted = torch.max(val_pred.data, 1)[1]
                            val_corr += (predicted == y_val).sum()

                        val_loss = self.criterion(val_pred, y_val)
                        self.logger.info(f"batch: {batch}/{len(self.train_loader)}, val loss: {val_loss.item()}")
                        self.writer.add_scalar("val_loss", val_loss.item(), global_step)
                        accuracy = int(float(val_corr) / (float(self.batch_size) * float(len(self.val_loader))) * 100.)
                        self.logger.info(f"val_accuracy: {accuracy}%")
                        self.writer.add_scalar("val_accuracy", accuracy, global_step)
                        batches = batch * self.batch_size
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                        save_model(self.model, epoch, batches, self.ckpt_dir, accuracy, best_accuracy, self.logger)

        self.logger.info(f"---------------- Training Finished ----------------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    args = parser.parse_args()

    # Get parameters from the config file
    with open(args.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    copy_config(config)

    Trainer = Train(config)
    Trainer.train()
