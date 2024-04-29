import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from collections import OrderedDict
from torch.backends import cudnn

from utils import draw_two_dimension

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.setup_seed(0)
        self.criterion = nn.MSELoss()
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Linear(1, 1)
        self.weight3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 20),
            'sig1': nn.Tanh(),
            'lin2': nn.Linear(20, 20),
            'sig2': nn.Tanh(),
            'lin3': nn.Linear(20, 20),
            'sig3': nn.Tanh(),
            'lin4': nn.Linear(20, 1)
        }))

    def forward(self, input1, input2, input3):
        output1 = self.weight1 * input1
        output2 = self.weight2(input2)
        output3 = self.weight3(input3)
        return output1, output2, output3

    def loss(self, x1, x2, x3, y1, y2, y3):
        self.eval()
        o1, o2, o3 = self.forward(x1, x2, x3)
        loss_1 = self.criterion(y1, o1)
        loss_2 = self.criterion(y2, o2)
        loss_3 = self.criterion(y3, o3)
        loss = loss_1 + loss_2 + loss_3
        return loss, [loss_1, loss_2, loss_3]

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True


def draw_loss(loss_list):
    plt.figure(figsize=(12, 9))
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.show()
    plt.close()


def train(model, args, x1, x2, x3, y1, y2, y3):
    device = x1.device
    model.train()
    model_save_path_last = "last.pt"
    print("using {}".format(device))
    print("epoch = {}".format(args.epoch))
    print("epoch_step = {}".format(args.epoch_step))
    print("model_save_path_last = {}".format(model_save_path_last))
    initial_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1/(x/10000+1))

    epoch_step = args.epoch_step
    start_time = time.time()
    start_time_0 = start_time
    loss_record = []
    loss1_record = []
    loss2_record = []
    loss3_record = []

    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        loss, loss_list = model.loss(x1, x2, x3, y1, y2, y3)
        loss.backward()
        optimizer.step()
        loss_record.append(float(loss.item()))
        loss1_record.append(float(loss_list[0].item()))
        loss2_record.append(float(loss_list[1].item()))
        loss3_record.append(float(loss_list[2].item()))
        if epoch % epoch_step == 0:
            now_time = time.time()
            loss_print_part = " ".join(["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
            print("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.6f}min in total, {7:.6f}min remains)".format(epoch, args.epoch, loss.item(), loss_print_part, optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)))
            start_time = time.time()
            torch.save({
                'epoch': args.epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss.item()}, model_save_path_last)
        scheduler.step()
        if epoch % args.save_step == 0 or epoch == args.epoch:
            draw_two_dimension(
                y_lists=[loss_record, loss1_record, loss2_record, loss3_record],
                x_list=range(1, 1 + len(loss_record)),
                color_list=["black", "r", "g", "b"],
                line_style_list=["solid"] * 4,
                legend_list=["loss", "loss1", "loss2", "loss3"],
                fig_y_label="loss"
            )
    return model


class Args:
    def __init__(self):
        self.epoch = 100000
        self.epoch_step = 10000
        self.lr = 0.01
        self.save_step = 100000


def run():
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"
    model = Net().to(device)
    x = torch.tensor([[i * 0.01] for i in range(100)], dtype=torch.float32).to(device)  # [[0.00], [0.01], ..., [0.99]]
    y1 = x * 3.1415926
    y2 = x * 5.0
    y3 = 100.0 * x * x + 10.0 * x + 10.0
    model = train(model, args, x, x, x, y1, y2, y3)
    # o1, o2, o3 = model(x, x, x)
    # print(o1.cpu().detach().numpy().flatten())
    # print(o2.cpu().detach().numpy().flatten())
    # print(o3.cpu().detach().numpy().flatten())
    print("weight1:", model.weight1.cpu().detach().numpy().flatten())
    print("weight2(0.1111):", model.weight2(torch.tensor([[0.1111111]]).to(device)).cpu().detach().numpy().flatten())
    print("weight3(0.9000):", model.weight3(torch.tensor([[0.9000000]]).to(device)).cpu().detach().numpy().flatten())


if __name__ == "__main__":
    run()
