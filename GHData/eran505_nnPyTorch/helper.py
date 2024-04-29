from collections import Counter

import pandas as pd
import os_util as pt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from torch_lr_finder import LRFinder
from torch import nn
from torch import optim
import csv


def log_file(list_items,to_path,column_names):
    df = pd.DataFrame(list_items, columns=column_names)
    df.to_csv(to_path,sep=',')

def plot_loss(path):
    father_p = '/'.join(str(path).split('/')[:-1])
    df = pd.read_csv(path,index_col=0)
    print(list(df))
    df["loss"].plot( kind='line')
    plt.savefig('{}/plot.png'.format(father_p))  # save the figure to file
    plt.show()

def value_statistic(path_to_df="/home/eranhe/car_model/generalization/4data/nn_DATA/all.csv"):
    df = pd.read_csv(path_to_df)
    matrix_df = df.to_numpy()
    size = len(matrix_df[:,-27:].flatten())
    c = Counter(matrix_df[:,-27:].flatten())
    xx= [(i, c[i] / size ) for i in c]
    sorted_by_second = sorted(xx, key=lambda tup: tup[1])
    reversed(sorted_by_second)
    for item in sorted_by_second:
        print("{},{}".format(item[0],item[1]))
    exit()

def concat_df():
    res = pt.walk_rec("/home/eranhe/car_model/generalization/4data/nn_DATA",[],".csv")
    l=[]
    print(res)
    for item in res:
        l.append(pd.read_csv(item))
    df_all = pd.concat(l)
    df_all.to_csv("/home/eranhe/car_model/generalization/4data/nn_DATA/all.csv",index=False)

def binn_polt(path_to_df="/home/eranhe/car_model/generalization/4data/nn_DATA/all.csv"):
    df = pd.read_csv(path_to_df)
    matrix_df = df.to_numpy()
    x = matrix_df[:, -27:].flatten()
    plt.hist(x, density=True, bins=30, label="Data")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 301)
    kde = st.gaussian_kde(x)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.legend(loc="upper left")
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.title("Histogram");
    plt.savefig("/home/eranhe/car_model/generalization/4data/nn_DATA/Histogram.png")
    plt.show()



def get_the_best_lr(my_model,my_criterion,trainloader,device='cpu'):
    my_optimizer = optim.Adam(my_model.parameters(), lr=1e-7, weight_decay=1e-1)
    lr_finder = LRFinder(my_model, my_optimizer, my_criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=10, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
#
# def lr_exp():
#     model = nn.Linear(2, 1)
#     optimizer = optim.SGD(model.parameters(), lr=0.5)
#     lr_scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
#     lr_scheduler_2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,20)
#     warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
#     warmup_scheduler.last_step = -1  # initialize the step counter
#
#     lrs = []
#
#     for i in range(100):
#         optimizer.step()
#         if i <= lr_scheduler_1.T_max:
#             lr_scheduler_1.step()
#         else:
#             lr_scheduler_2.step()
#         lrs.append(
#             optimizer.param_groups[0]["lr"]
#         )
#         warmup_scheduler.dampen()
#
#     plt.plot(lrs)
#     plt.show()

# def other():
    # model = nn.Linear(2, 1)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    # num_steps = 100
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=0.00001,last_epoch=-1)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # lrs = []
    # for epoch in range(1, 100 + 1):
    #     optimizer.step()
    #     lr_scheduler.step()
    #     warmup_scheduler.dampen()
    #     lrs.append(optimizer.param_groups[0]["lr"])
    # plt.plot(lrs)
    # plt.show()

def process_path(path_str):
    arr = [eval(x[1:-2]) for x in path_str if len(x) > 5]
    return arr
def load__p(p_path):
    l=[]
    with open(p_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            if len(line) < 1:
                continue
            d = {'p': float(str(line[0]).split(':')[-1]), 'traj': process_path(line[1:])}
            l.append(d)
    return l

def diff_pz(p1,p2):
    l1 = load__p(p1)
    l2 = load__p(p2)

    print(l1)
    ctr=0
    ctr_not=0
    for item in l1:
        for pos in item["traj"]:
            bol=False
            for item_other in l2:
                if bol:
                    break
                for pos_other in item_other["traj"][10:-2]:
                    if pos[0]==pos_other[0]:
                        ctr+=1
                        bol=True
                        break

            if bol is False:
                ctr_not+=1

    print("yes: {}   not:{}   sum: {}".format(ctr/(ctr+ctr_not),ctr_not/(ctr+ctr_not),ctr+ctr_not))
    exit()

if __name__ == "__main__":
    diff_pz("/home/ERANHER/car_model/generalization/data/p.csv","/home/ERANHER/car_model/generalization/12data/p.csv")
    value_statistic()
    plot_loss("/home/ERANHER/car_model/nn/loss_train.csv")
    pass