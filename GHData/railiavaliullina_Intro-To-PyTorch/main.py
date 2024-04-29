import torch
import torch.autograd.profiler as profiler
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
import mlflow
import numpy as np
import time
import os
import warnings

from config import cfg
from model import LogisticRegression, MLP1HL, MLP2HL
from data import get_data
warnings.filterwarnings('ignore')


"""
УЛУЧШЕНИЯ:

1) вынести в отдельную папку
2) создать класс, вынести туда функции для обучения, инициализации модели, оптимайзера, лосса и т.д. -> вынести в model.py
3) добавить документацию к каждой функции, следить за названиями функций
4) TODO: ...
5) минимизировать количество кода в main.py (только вызывать функции из других файлов)
"""


def get_criterion():
    criterion_ = torch.nn.CrossEntropyLoss()
    return criterion_


def get_optimizer(model_):
    optimizer_ = torch.optim.SGD(model_.parameters(), lr=cfg.lr)
    return optimizer_


def make_step(model_, optimizer_, criterion_, images, labels, global_step):
    optimizer_.zero_grad()
    images = images.view(-1, cfg.input_dim)
    if cfg.use_profiler:
        with profiler.profile(record_shapes=True) as prof:
            logits = model_(images)
        print(prof.key_averages().table())
        prof.export_chrome_trace(f'trace.json')
    else:
        logits = model_(images)
    cross_entropy_loss = criterion_(logits, labels)
    l2_reg = torch.tensor(0.0, requires_grad=True)
    for p in model_.parameters():
        l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
    loss = cross_entropy_loss + l2_reg
    loss.backward()
    optimizer_.step()
    if cfg.log_metrics:
        # TODO: для компактности можно вынести в utils.py функцию логирования, на вход которой подавать
        #  dict (name: value); внутри фцнкции идти по ним и логировать
        mlflow.log_metric('train/cross_entropy_loss', cross_entropy_loss.item(), global_step)
        mlflow.log_metric('train/reg_loss', l2_reg.item(), global_step)
        mlflow.log_metric('train/total_loss', loss.item(), global_step)
    return loss.item(), l2_reg.item(), cross_entropy_loss.item()


# TODO: вынести в отдельный файл все, что связано с валидацией
def evaluate(model_, dl, epoch, set_type):
    print(f'evaluating on {set_type} data...')
    eval_start_time = time.time()
    correct, total = 0, 0
    cross_entropy_losses, reg_losses, losses = [], [], []
    unique_labels = np.unique(dl.dataset.train_labels) if set_type is 'train' else np.unique(dl.dataset.test_labels)
    accuracies_for_classes = [0 for _ in unique_labels]
    for images, labels in dl:
        images = images.view(-1, cfg.input_dim)
        logits = model_(images)
        _, predicted = torch.max(logits.data, 1)
        predicted = predicted.cpu()
        total += labels.size(0)
        correct += torch.sum(predicted == labels)

        for i, l in enumerate(labels):
            accuracies_for_classes[l] += torch.sum((predicted[i] == l))

        if set_type is 'test':  # calculate losses
            cross_entropy_loss = criterion(logits, labels)
            cross_entropy_losses.append(cross_entropy_loss.item())
            l2_reg = torch.tensor(0.0, requires_grad=True)
            for p in model.parameters():
                l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
            reg_losses.append(l2_reg.item())
            losses.append((cross_entropy_loss + l2_reg).item())

    if set_type is 'test' and cfg.log_metrics:
        mlflow.log_metric('test/cross_entropy_loss', np.mean(cross_entropy_losses), epoch)
        mlflow.log_metric('test/reg_loss_train', np.mean(reg_losses), epoch)
        mlflow.log_metric('test/total_loss_train', np.mean(losses), epoch)

    accuracy = 100 * correct.item() / total
    print(f'Accuracy on {set_type} data: {accuracy}')
    accuracies_for_classes = [100 * acc.item()/dl.dataset.nb_images_per_class[i] for i, acc in enumerate(accuracies_for_classes)]
    print(f'accuracies for classes: {accuracies_for_classes}')
    balanced_acc = sum(accuracies_for_classes)/len(dl.dataset.classes)
    print(f'Balanced accuracy: {balanced_acc}')

    if cfg.log_metrics:
        mlflow.log_metric(f'{set_type}/accuracy', accuracy, epoch)
        for i, acc in enumerate(accuracies_for_classes):
            mlflow.log_metric(f'{set_type}/accuracy_class_{i}', acc, epoch)
        mlflow.log_metric(f'{set_type}/balanced_accuracy', balanced_acc, epoch)
    print(f'evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')


def train(cfg_, model_, dl_train_, dl_test_):
    global_step = 0
    for e in range(cfg_.epochs):
        losses, reg_losses, cross_entropy_losses = [], [], []
        epoch_start_time = time.time()
        print(f'Epoch: {e}')
        for i, (images, labels) in enumerate(dl_train_):
            loss, reg_loss, cross_entropy_loss = make_step(model_, optimizer, criterion, images, labels, global_step)
            losses.append(loss)
            reg_losses.append(reg_loss)
            cross_entropy_losses.append(cross_entropy_loss)
            global_step += 1

            if global_step % 100 == 0:
                mean_loss = np.mean(losses[:-20]) if len(losses) > 20 else np.mean(losses)
                mean_reg_loss = np.mean(reg_losses[:-20]) if len(reg_losses) > 20 else np.mean(reg_losses)
                mean_cross_entropy_loss = np.mean(cross_entropy_losses[:-20]) if len(cross_entropy_losses) > 20 \
                    else np.mean(cross_entropy_losses)
                print(f'step: {global_step}, total_loss: {mean_loss}, cross_entropy_loss: {mean_cross_entropy_loss}, '
                      f'reg_loss: {mean_reg_loss}')
        if cfg['logging']['log_metrics']:
            mlflow.log_metric('train/loss_mean', np.mean(losses), e)
            mlflow.log_metric('train/reg_loss_mean', np.mean(reg_losses), e)
            mlflow.log_metric('train/cross_entropy_loss_mean', np.mean(cross_entropy_losses), e)

        model_.eval()
        with torch.no_grad():
            for dl, set_type in zip([dl_test_, dl_train_], ['test', 'train']):
                evaluate(model_, dl, e, set_type)

        model_.train()
        print(f'epoch training time: {round((time.time() - epoch_start_time)/60, 3)} min')
        if e % 5 == 0:
            print('Saving current model...')
            state = {
                'model': model_.state_dict(),
                'epoch': e,
                'global_step': global_step,
                'opt': optimizer.state_dict(),
            }
            torch.save(state, (os.path.join(cfg.checkpoints_dir, f'checkpoint_{e}.pth')))


if __name__ == '__main__':
    # TODO: добавить опционально обучение на GPU, CPU;
    dl_train, dl_test = get_data()
    model = globals()[cfg.model](cfg.input_dim, cfg.output_dim)

    if cfg.model is 'LogisticRegression':
        model = LogisticRegression(cfg.input_dim, cfg.output_dim)
    elif cfg.model is 'MLP1HL':
        model = MLP1HL(cfg.input_dim, cfg.output_dim)
    elif cfg.model is 'MLP2HL':
        model = MLP2HL(cfg.input_dim, cfg.output_dim)
    else:
        raise Exception

    print('\nModel parameters: ')
    print(model)

    criterion = get_criterion()
    optimizer = get_optimizer(model)

    nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters number: {nb_trainable_params}')

    start_time = time.time()
    train(cfg, model, dl_train, dl_test)
    print(f'Total training time: {round((time.time() - start_time)/60, 3)} min')
