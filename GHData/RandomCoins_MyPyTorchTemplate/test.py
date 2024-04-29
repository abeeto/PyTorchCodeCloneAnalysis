import argparse
from dataset import TestSet
from torch.utils.data import DataLoader
import torch
from model import Model
from utils import AverageMeter, get_remain_time
import os
import time
import pandas as pd
from scipy.stats import spearmanr


def test(model, dataloader, device):
    model.eval()
    time_meter = AverageMeter()

    predict = []
    labels = []
    filename = []

    with torch.no_grad():
        for i, (image, label, name) in enumerate(dataloader):
            start = time.time()
            image = image.to(device)
            output = model(image)
            predict.append(output.detach().cpu().item())
            labels.append(label.detach().cpu().item())

            time_meter.update(time.time() - start)
            remain_time = get_remain_time(i, len(dataloader), time_meter.avg)
            print('\r{}: {}/{} [remain: {}]'.format(test.__name__, i + 1, len(dataloader), remain_time), end='',
                  flush=True)
        print()
        corr, _ = spearmanr(predict, labels)
        df = pd.DataFrame({'image': filename, 'predict': predict, 'label': labels})
        df.to_excel('result.xlsx', index=False)
        print('Test - SROCC: {:.4f}'.format(corr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', type=str)
    parser.add_argument('-model', '--model', type=str)
    parser.add_argument('-gpu', '--gpu', type=str)

    cfg = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    dataset = TestSet(cfg.data)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    print('==> Loading model...')
    if os.path.isfile(cfg.model) and os.path.exists(cfg.model):
        model = Model()
        model.load_state_dict(torch.load(cfg.model, map_location='cpu')['model'])
    else:
        raise Exception(f'The model {cfg.model} not exist')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    test(model, dataloader, device)
