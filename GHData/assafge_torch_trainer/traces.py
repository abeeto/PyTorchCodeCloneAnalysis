try:
    from .general_utils import plot_confusion_matrix
except ImportError:
    from general_utils import plot_confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import abc
import random


class Trace(metaclass=abc.ABCMeta):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        self.writer = writer
        self.mini_batches = mini_batches

    @abc.abstractmethod
    def add_measurement(self, inputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        pass

    @abc.abstractmethod
    def write_epoch(self, epoch: int):
        pass

class StepTrace(Trace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.sum = 0.0

    def write_epoch(self, epoch: int):
        self.writer.add_scalar(tag=self.__class__.__name__, scalar_value=self.sum / self.mini_batches, global_step=epoch)
        self.sum = 0.0


class PixelWiseAccuracy(StepTrace):
    def add_measurement(self, inputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        self.sum += (int(torch.sum(predictions.argmax(dim=1) == labels.data).to('cpu')) / labels.data.nelement()) * 100


class QuaziPixelWiseAccuracy(PixelWiseAccuracy):
    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        super().add_measurement(predictions, labels)
        super().add_measurement(predictions, labels + 1)
        super().add_measurement(predictions, labels - 1)


class ImageTrace(Trace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.did_wrote = False  # write every image once
        self.step = 0
        self.inp: np.ndarray = None
        self.pred: np.ndarray = None
        self.lbl: np.ndarray = None
        self.channels = 3

    def add_measurement(self, inputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        if not self.did_wrote:
            rand_ind = int(random.random() * predictions.shape[0])
            self.inp = (inputs[rand_ind].cpu().numpy() * 255).astype(np.uint8)
            self.pred = (predictions[rand_ind].cpu().numpy() * 255).astype(np.uint8)
            self.lbl = (labels[rand_ind].cpu().numpy() * 255).astype(np.uint8)
            self.did_wrote = True

    def write_epoch(self, step):
        self.did_wrote = False
        if self.channels == 1:
            h = self.inp.shape[0]
            w = self.inp.shape[1]
        else:
            h = self.inp.shape[1]
            w = self.inp.shape[2]
        out = np.zeros((3, h, 3 * w), dtype=np.uint8)
        out[:, :, :w] = self.inp
        out[:, :, w:2*w] = self.pred[:3, :, :]
        out[:, :, 2*w:] = self.lbl[:3, :, :]
        self.writer.add_image('input | predicted | label', out, global_step=step, dataformats='CHW')
        self.step = step

class DepthImageTrace(ImageTrace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.channels = 1

    def add_measurement(self, inputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        if not self.did_wrote:
            rand_ind = int(random.random() * predictions.shape[0])
            inp = inputs[rand_ind].cpu().numpy()
            im = predictions[rand_ind].cpu().numpy()
            lbl = labels[rand_ind].cpu().numpy()
            self.inp = (inp * 255).astype(np.uint8)
            self.pred = ((im + 5) * (255/15)).astype(np.uint8)
            self.lbl = ((lbl + 5) * (255/15)).astype(np.uint8)
            self.did_wrote = True

class ClassificationImageTrace(ImageTrace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.channels = 1

    def add_measurement(self, inputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        if not self.did_wrote:
            classes = predictions.shape[1]
            rand_ind = int(random.random() * predictions.shape[0])
            self.pred = (predictions[rand_ind].argmax(dim=0).cpu().numpy() * (255 / classes)).astype(np.uint8)
            self.lbl = (labels[rand_ind].cpu().numpy() * (255 / classes)).astype(np.uint8)
            self.did_wrote = True

class ConfusionMatrix(ImageTrace):
    def add_measurement(self, inputs, outputs, labels):
        """inputs not in use (match interface)"""
        if not self.did_wrote:
            pred = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8).ravel()
            ref = labels.cpu().numpy().astype(np.uint8).ravel()
            im = plot_confusion_matrix(ref, pred, [str(i) for i in range(16)])
            self.writer.add_image('confusion_mat', im, global_step=self.step, dataformats='HWC')
            self.did_wrote = True