import torch
import sys
import time
from metric import Metric


def train(epoch, dataloader, device, optimizer, model, criterion, scheduler, tb_writer):
    model.train()
    metric = Metric()
    start_time = time.time()

    for step, dataset in enumerate(dataloader):
        # data
        input_ids = dataset['inputs'].to(device)
        attention_mask = dataset['mask'].to(device)
        labels = dataset['labels'].type(torch.FloatTensor).to(device)

        # train model
        optimizer.zero_grad()
        model.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs[0])
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        metric.update(preds.clone().detach().cpu().numpy(), labels.clone().detach().cpu().numpy(), loss.item())
        print('step {}/{}, loss : {}'.format(step + 1, len(dataloader), loss.item()), flush=True)
        sys.stdout.write("\033[F")

        del loss
        del preds
        del outputs

    train_result = {k: v for k, v in metric.items() if k in ['loss', 'f1', 'precision', 'recall', 'fpr']}
    train_result['phase'] = 'train'
    print('Epoch {} execution time {} (sec)'.format(epoch+1, time.time() - start_time))
    print('[{} Epoch] train result'.format(epoch+1))
    print(train_result)

    tb_writer.add_scalar("{}/{}".format('train', 'Loss'), metric['loss'], epoch)
    tb_writer.add_scalar("{}/{}".format('train', 'Precision'), metric['precision'], epoch)
    tb_writer.add_scalar("{}/{}".format('train', 'Recall'), metric['recall'], epoch)
    tb_writer.add_scalar("{}/{}".format('train', 'F1'), metric['f1'], epoch)
    tb_writer.add_scalar("{}/{}".format('train', 'FPR'), metric['fpr'], epoch)

    return train_result


def eval(epoch, dataloader, device, model, criterion, tb_writer, is_test=False):
    model.eval()
    metric = Metric()
    start_time = time.time()

    with torch.no_grad():
        for step, dataset in enumerate(dataloader):
            input_ids = dataset['inputs'].to(device)
            attention_mask = dataset['mask'].to(device)
            labels = dataset['labels'].type(torch.FloatTensor).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs[0])

            v_loss = criterion(preds, labels)
            metric.update(preds.clone().detach().cpu().numpy(), labels.clone().detach().cpu().numpy(),
                          v_loss.item())

            del v_loss
            del preds
            del outputs

    phase = 'test' if is_test else 'val'

    eval_result = {k: v for k, v in metric.items() if k in ['loss', 'f1', 'precision', 'recall', 'fpr']}
    eval_result['phase'] = phase
    print('Epoch {} execution time {} (sec)'.format(epoch+1, time.time() - start_time))
    print('[{} Epoch] {} result'.format(epoch+1, phase))
    print(eval_result)

    tb_writer.add_scalar("{}/{}".format(phase, 'Loss'), metric['loss'], epoch)
    tb_writer.add_scalar("{}/{}".format(phase, 'Precision'), metric['precision'], epoch)
    tb_writer.add_scalar("{}/{}".format(phase, 'Recall'), metric['recall'], epoch)
    tb_writer.add_scalar("{}/{}".format(phase, 'F1'), metric['f1'], epoch)
    tb_writer.add_scalar("{}/{}".format(phase, 'FPR'), metric['fpr'], epoch)

    return eval_result
