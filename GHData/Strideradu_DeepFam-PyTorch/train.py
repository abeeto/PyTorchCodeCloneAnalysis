import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel

from dataset import *
from utils import *
from models import *
from sklearn.metrics import roc_auc_score
import tqdm


def train(train_loader, val_loader, model, optimizer, args):
    model.cuda()

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        print("Epoch {}/{}".format(epoch, args.epochs))
        losses = []
        for batch in tqdm.tqdm(train_loader):
            feature, target = batch[0], batch[1]
            # print(feature.shape)
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = data_parallel(model, feature)

            # print('prob vector', prob.size())
            # print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        accuracy, val_losses, corrects, size = eval(val_loader, model)
        print('\nTrain - loss: {:.6f} Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(np.mean(losses),
                                                                                              np.mean(val_losses),
                                                                                              accuracy, corrects, size))
        if accuracy > best_acc:
            best_acc = accuracy
            save_checkpoint(args.checkpoint_path, model, optimizer)

    print("Best accuracy is {:.4f}".format(best_acc))


def eval(data_loader, model):
    model.eval()
    corrects = 0
    losses = []
    for batch in data_loader:
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()

        logit= model(feature)
        loss = F.cross_entropy(logit, target)

        losses.append(loss.item())
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_loader.dataset)
    accuracy = 100 * corrects.data.cpu().numpy() / size
    return accuracy, losses, corrects, size


if __name__ == '__main__':
    args = argparser()

    train_data = PepseqDataset(file_path=args.train_file)
    # test_data = PepseqDataset(file_path="/home/dunan/Documents/DeepFam_data/GPCR/cv_1/test.txt")
    test_data = PepseqDataset(file_path=args.test_file)

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

    args = argparser()
    model = PepCNN()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.regularizer)

    train(train_loader, test_loader, model, optimizer, args)
