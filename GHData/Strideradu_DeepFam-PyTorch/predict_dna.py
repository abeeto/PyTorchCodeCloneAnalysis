import torch
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel
from dataset import *
from utils import *
from models import *
import os
import tqdm
from apex import amp


def load_checkpoint(checkpoint_path, model, optimizer=None, gpu=True):
    if gpu:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def predict(args):
    model = PepCNN(num_class=args.num_classes)
    load_checkpoint(args.checkpoint_path, model)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters())
    model, _ = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model.eval()

    predict_data = PepseqDatasetFromDNA(args.test_file)
    data_loader = data.DataLoader(predict_data, batch_size=args.batch_size)

    corrects = 0
    for batch in tqdm.tqdm(data_loader):
        feature1, feature2, feature3, target = batch[0], batch[1], batch[2], batch[3]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        feature1, feature2, feature3, target = feature1.cuda(), feature2.cuda(), feature3.cuda(), target.cuda()

        # logit = data_parallel(model, feature)
        logit1 = model(feature1)
        prob1 = F.softmax(logit1, 1)

        scores1, preds1 = torch.max(prob1, 1)
        scores1, preds1 = scores1.tolist(), preds1.tolist()

        logit2 = model(feature2)
        prob2 = F.softmax(logit2, 1)

        scores2, preds2 = torch.max(prob2, 1)
        scores2, preds2 = scores2.tolist(), preds2.tolist()

        logit3 = model(feature3)
        prob3 = F.softmax(logit3, 1)

        scores3, preds3 = torch.max(prob3, 1)
        scores3, preds3 = scores3.tolist(), preds3.tolist()

        targets = target.tolist()
        for i, pred in enumerate(preds1):
            if pred == targets[i] or preds2[i] == targets[i] or preds3[i] == targets[i]:
                corrects += 1

    size = len(data_loader.dataset)
    accuracy = 100 * corrects / size
    print("acc: {:.4f}%({}/{})".format(accuracy, corrects, size))


if __name__ == '__main__':
    args = argparser()
    # assert args.predict_file is not None
    predict(args)
