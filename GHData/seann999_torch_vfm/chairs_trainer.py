import torch
from tensorboard_logger import configure, log_value
from vfm import VFM
import chairs_loader as chairs
from chairs_loader import ChairsDataset
import argparse
import os
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='vfm')
parser.add_argument('--model', type=str, default="runs/A", metavar='G',
                    help='model path')
args = parser.parse_args()

configure(args.model, flush_secs=5)
batch_size = 32

def train_chairs():
    train_loader = torch.utils.data.DataLoader(
        ChairsDataset(chairs.train_chairs),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=False, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        ChairsDataset(chairs.test_chairs),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True)

    model = VFM().cuda()
    epoch = 0

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    # model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # hidden = model.init_hidden(args.batch_size)

    while True:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict()
        }, '{}/checkpoint.tar'.format(args.model))

        def run(train):
            total_loss = 0

            if train:
                title = "train"
            else:
                title = "test"

            if train:
                loader = train_loader
            else:
                loader = test_loader

            for i, (mat1, act, mat2) in enumerate(loader):
                x = Variable(mat1).cuda()
                y = Variable(mat2).cuda()

                acts = Variable(act).cuda()

                # hidden = repackage_hidden(hidden)
                optimizer.zero_grad()
                recon, mean, logvar = model.forward(x, acts)

                #if train:
                loss = model.loss(y, recon, mean, logvar)
                total_loss += loss.data[0]
                loss.backward()

                if train:
                    optimizer.step()

                x_img = np.hstack([np.moveaxis(x.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])
                y_img = np.hstack([np.moveaxis(y.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])
                pred_img = np.hstack([np.moveaxis(recon.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])

                cv2.imshow(title, np.vstack([x_img, y_img, pred_img]))
                cv2.waitKey(1)

                if train:
                    log_value("{} loss".format(title), loss.data[0], i + epoch * len(train_loader))

                    print("epoch {}, step {}/{}: {}".format(epoch, i, len(train_loader), loss.data[0]))

                print(act.numpy()[0])


            epoch_loss = total_loss / len(loader)
            log_value('epoch {} loss'.format(title), epoch_loss, epoch)
            print("AVG LOSS: {}".format(epoch_loss))

        run(True)
        run(False)
        epoch += 1

def lerp_chairs():
    global batch_size
    batch_size = 1
    chairs.steps = 1

    z_size = 32
    model = VFM().cuda()

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    while True:
        print("generating")
        lerps = []
        start = np.random.uniform(-2, 2, (1, z_size))
        end = np.random.uniform(-2, 2, (1, z_size))

        for x in np.linspace(0, 1, 10):
            gen = model.decode(Variable(torch.FloatTensor(end*x+start*(1-x))).cuda())
            lerps.append(np.moveaxis(gen.data.cpu().numpy()[0, 0, ...], 0, -1))

        cv2.imshow("lerp", np.hstack(lerps))
        cv2.waitKey(0)
        #cv2.imwrite("image-{}.png".format(k), np.hstack(lerps))
        print("generated ")

if __name__ == "__main__":
    #train_chairs()
    lerp_chairs()