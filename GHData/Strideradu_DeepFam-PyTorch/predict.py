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
    load_checkpoint(args.checkpoint_path, model, gpu=False)
    # model.cuda()
    # optimizer = torch.optim.Adam(params=model.parameters())
    # model, _ = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model.eval()

    probs = []
    topps = []
    topks = []

    predict_data = PepseqDataset(args.test_file)
    data_loader = data.DataLoader(predict_data, batch_size=args.batch_size)

    corrects = 0
    for batch in tqdm.tqdm(data_loader):
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # feature, target = feature.cuda(), target.cuda()

        # logit = data_parallel(model, feature)
        logit = model(feature)
        prob = F.softmax(logit, 1)

        probs.append(prob.data.cpu().numpy())

        corrects += (torch.max(prob, 1)
                     [1].view(target.size()).data == target.data).sum()
        logit_5, top5 = torch.topk(prob.data.cpu(), args.topk)
        for i, l in enumerate(logit_5):
            topps.append(l.numpy())
            topks.append(top5[i].numpy())

    size = len(data_loader.dataset)
    accuracy = 100 * corrects.data.cpu().numpy() / size
    print("acc: {:.4f}%({}/{})".format(accuracy, corrects, size))

    """

    if args.predict_file:
        df = pd.read_csv(args.test_file, sep='\t', header=None)
        df["topps"] = topps
        df["topk"] = topks
        df.to_csv(args.predict_file, columns=[2, 0, "topk", "topps"])

    probs = np.concatenate(probs)
    parent_dir = os.path.dirname(args.predict_file)
    filename_base = os.path.splitext(os.path.basename(args.predict_file))[0]
    prob_path = os.path.join(parent_dir, filename_base+'.npy')
    np.save(prob_path, probs)
    """


if __name__ == '__main__':
    args = argparser()
    # assert args.predict_file is not None
    predict(args)
