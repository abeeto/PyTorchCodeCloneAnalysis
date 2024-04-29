import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from models import ProbUNet
from dataset import LeafDataset

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class Trainer:

    def __init__(self, epochs=100, batch_size=32, lr=1e-4, use_gpu=True):

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        torch.autograd.set_detect_anomaly(True)
        self.net = ProbUNet(latent_dim=4,
                            in_channels=3,
                            num_classes=1,
                            num_1x1_convs=3,
                            init_features=16).to(self.device)


        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)
        # self.opt = torch.optim.Adam([
        #     {"params": self.net._unet.parameters(),
        #      "lr": self.lr * 10},
        #     {"params": self.net._f_comb.parameters(),
        #      "lr": self.lr * 1},
        #     {"params": self.net._posterior.parameters(),
        #      "lr": self.lr}
        # ])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', factor=0.1, patience=10)

        train_dataset = LeafDataset('data/CVPPP2014_LSC_training_data')
        val_dataset = LeafDataset('data/CVPPP2014_LSC_validation_data')

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


    def train(self):

        train_iteration, val_iteration = 0, 0
        for epoch in range(self.epochs):

            self.net.train()
            for i, (img_batch, fg_batch) in enumerate(self.train_dataloader):

                # img_batch: 1 x 3 x H x W
                # fg_batch: 1 x N x H x W
                # Each of the N masks should be combined with the img and passed to network
                c, h, w = img_batch.shape[1:]
                n = fg_batch.shape[1]
                fg_batch = fg_batch.squeeze(0).unsqueeze(1)  # N x 1 x H x W
                img_batch = img_batch * torch.ones(n, c, h, w, dtype=torch.float32)  # N x 3 x H x W

                img_batch, fg_batch = next(iter(DataLoader(TensorDataset(img_batch, fg_batch), n)))

                img_batch = img_batch.to(self.device)
                fg_batch = fg_batch.to(self.device)

                pred = self.net(img_batch, fg_batch)
                loss = self.net.elbo(pred, fg_batch)
                self.net.zero_grad()

                loss.backward()
                self.opt.step()

                print("train loss", loss.detach().data.cpu())

                train_iteration += 1



            if train_iteration % 10 == 0:
                self.net.eval()
                with torch.no_grad():
                    for i, (img_batch, fg_batch) in enumerate(self.val_dataloader):
                        # img_batch: 1 x 3 x H x W
                        # fg_batch: 1 x N x H x W
                        # Each of the N masks should be combined with the img, and passed to network
                        c, h, w = img_batch.shape[1:]
                        n = fg_batch.shape[1]
                        fg_batch = fg_batch.squeeze(0).unsqueeze(1)  # N x 1 x H x W
                        img_batch = img_batch * torch.ones(n, c, h, w, dtype=torch.float32)  # N x 3 x H x W

                        img_batch, fg_batch = next(iter(DataLoader(TensorDataset(img_batch, fg_batch), n)))

                        img_batch = img_batch.to(self.device)
                        fg_batch = fg_batch.to(self.device)

                        pred = self.net(img_batch, fg_batch)
                        loss = self.net.elbo(pred, fg_batch)

                        print("val loss", loss.detach().data.cpu())


                        val_iteration += 1

        self.save_model()

    def save_model(self):
        torch.save(self.net.state_dict(), 'prob_unet.pth')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_gpu', action='store_false')
    args = parser.parse_args()
    runner = Trainer(args.epochs, args.batch_size, args.lr, args.use_gpu)
    runner.train()

