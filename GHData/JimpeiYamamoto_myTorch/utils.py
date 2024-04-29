import os
import glob
import shutil
import random
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, roc_auc_score

def imagepath_to_tu(path, df, img_row_name='img_path', tu_row_name='treated_t'):
    '''
    path: 画像のpath
    df: 水質データのdataframe
    img_row_name: df内の画像の列名
    tu_row_name: df内の処理水濁度の列名
    '''
    file_name = os.path.basename(path)
    col = df[df[img_row_name] == file_name]
    return float(col[tu_row_name])

def split_class_image(srcs, df, dest0, dest1, dest2):
    '''
    srcs: ソース
    df: 水質データが入ったdataframe
    dest0: < 0.5の画像を保存するpath
    dest1: <= 1.0の画像を保存するpath
    dest2: > 1.0の画像を保存するpath
    '''
    srcs_files = glob.glob(os.path.join(srcs, "*.jpg"))
    for file in srcs_files:
        tu = imagepath_to_tu(file, df)
        if tu < 0.5:
            shutil.move(file, os.path.join(dest0, os.path.basename(file)))
        elif tu <= 1.0:
            shutil.move(file, os.path.join(dest1, os.path.basename(file)))
        else:
            shutil.move(file, os.path.join(dest2, os.path.basename(file)))

def split_train_validation(dest, srcs, vali_rate):
    '''
    dest: サンプリングしたvalidationファイルの移動先のフォルダ
    srcs: サンプリング前のファイルが保存されているフォルダ
    vali_rate: validationファイルの比率
    '''
    srcs_files = glob.glob(srcs)
    file_len = int(len(srcs_files) * vali_rate)
    for _ in range(file_len):
        srcs_files = glob.glob(srcs)
        file = random.sample(srcs_files, 1)[0]
        shutil.move(file, os.path.join(dest, os.path.basename(file)))

def test(net, test_loader, heatmap_path, device, criterion):
    '''
    net: 学習済みモデル
    test_loader: test_loader
    heatmap_path: 作成したヒートマップの保存パス
    device: device
    criterion: 最適化関数
    '''
    test_loss = 0.0
    test_acc = 0.0
    net.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        test_loss =  loss.item() / len(test_loader.dataset)
        test_acc = (outputs.max(1)[1] == labels).sum() / len(test_loader.dataset)
    print("test_loss: {0:4f}, test_acc: {1:4f}".format(test_loss, test_acc))
    cm = confusion_matrix(labels, outputs.argmax(1))
    sns.heatmap(cm)
    plt.savefig(heatmap_path)
    return outputs

def split_correct_image_csv(test_df, outcome_df_path, correct_path, incorrect_path, custom_test_dataset, outputs, labels):
    '''
    correct_path: 正解だった画像の保存先
    incorrect_path: 不正解だった画像の保存先
    custom_test_dataset: テストカスタムデータセット
    outputs: テストの結果
    labels: テストデータのラベル
    '''
    correct_lst = []
    for i, image_path in enumerate(custom_test_dataset.images):
        output = outputs[i].argmax()
        file_name = os.path.basename(image_path)
        if output == labels[i]:
            dst = correct_path
            correct_lst.append(1)
        else:
            dst = incorrect_path
            correct_lst.append(0)
        shutil.copyfile(image_path, os.path.join(dst, file_name))
    df = test_df.copy()
    df['label'] = labels
    df['outputs'] = outputs.argmax(1)
    df['iscorrect'] = correct_lst
    df.to_csv(outcome_df_path)

def plot_epoch_to_loss(num_epoch, train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label = 'train_loss')
    plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label = 'val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()


def plot_epoch_to_loss_acc(num_epoch, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    plt.figure()
    plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label = 'train_loss')
    plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label = 'val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(num_epoch), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(num_epoch), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation acc')
    plt.grid()

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def cnt_each_class_labels(dataset):
    cnt_0 = 0
    cnt_1 = 0
    cnt_2 = 0
    for label in dataset.labels:
        if label == 0:
            cnt_0+=1
        elif label == 1:
            cnt_1+=1
        else:
            cnt_2+=1
    print("class0: ", cnt_0)
    print("class1: ", cnt_1)
    print("class2: ", cnt_2)

def plot_matrix(labels_lst, outputs_lst):
    cm = confusion_matrix(labels_lst, outputs_lst)
    sns.heatmap(cm)
    print(cm)

def plot_hist_line(correct_lst, incorrect_lst, min_, max_, bandwidth):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(correct_lst, alpha=0.3, label="correct")
    ax1.hist(incorrect_lst, alpha=0.3, label="error")
    ax1.set_xlabel("model uncertainty")
    ax1.set_ylabel("density")
    plt.legend()
    ax2 = ax1.twinx()
    x = np.linspace(min_, max_, 2000)[:,None]
    df_co = pd.DataFrame()
    df_co['correct'] = correct_lst
    kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(df_co['correct'][:,None]))
    dens1 = kde1.score_samples(x)
    ax2.plot(x, np.exp(dens1))
    df_in = pd.DataFrame()
    df_in['incorrect'] = incorrect_lst
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(df_in['incorrect'][:,None]))
    dens2 = kde2.score_samples(x)
    ax2.plot(x, np.exp(dens2))

def plt_roc_auc(outcome_lst, model_uncertainty_lst, data_uncertainty_lst):
    fpr, tpr, _ = roc_curve(outcome_lst, model_uncertainty_lst)
    plt.plot(fpr, tpr, label='model uncertainty')
    plt.fill_between(fpr, tpr, 0, alpha=0.3)
    fpr, tpr, _ = roc_curve(outcome_lst, data_uncertainty_lst)
    plt.plot(fpr, tpr, label='data uncertainty')
    plt.fill_between(fpr, tpr, 0, alpha=0.3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
    print(f'model uncertaity AUC: {roc_auc_score(outcome_lst, model_uncertainty_lst):.4f}')
    print(f'data uncertainty AUC: {roc_auc_score(outcome_lst, data_uncertainty_lst):.4f}')

def under_sample_alt_class(info, df):
    num_class = info[0].shape[0]
    num_range = info[0].shape[1]
    min_cnt = info[0][0][0]
    for i in range(num_class):
        for j in range(num_range):
            min_cnt = min([min_cnt, info[0][i][j]])
    df_train_equal = pd.DataFrame()
    for i in range(len(info[1]) - 1):
        start = info[1][i]
        stop = info[1][i+1]
        df0 = df.copy()[df['class0'].copy() == 1]
        df1 = df.copy()[df['class1'].copy() == 1]
        df2 = df.copy()[df['class2'].copy() == 1]
        tmp_df0 = df0.copy()[(df0['ALT'] > start) & (df0['ALT'] <= stop)].sample(n = int(min_cnt)-2)
        tmp_df1 = df1.copy()[(df1['ALT'] > start) & (df1['ALT'] <= stop)].sample(n = int(min_cnt)-2)
        tmp_df2 = df2.copy()[(df2['ALT'] > start) & (df2['ALT'] <= stop)].sample(n = int(min_cnt)-2)
        df_train_equal = pd.concat([df_train_equal.copy(), tmp_df0, tmp_df1, tmp_df2])
    return df_train_equal


def hist_class(df, col_name, bins, rang):
    if rang[0]==rang[1]:
        info = plt.hist(
            [
                df[col_name][df['class0']==1],
                df[col_name][df['class1']==1],
                df[col_name][df['class2']==1]
            ],
            bins=bins,
            stacked=True,
            label=['class0', 'class1', 'class2'],
        )
    else:
        info = plt.hist(
            [
                df[col_name][df['class0']==1],
                df[col_name][df['class1']==1],
                df[col_name][df['class2']==1]
            ],
            bins=bins,
            stacked=True,
            label=['class0', 'class1', 'class2'],
            range=rang
        )
    plt.legend()
    plt.grid()
    for i in range(len(info[1]) - 1):
        print(
            "[{:7} ~ {:7}] ".format(int(info[1][i] * 100) / 100, int(info[1][i+1] * 100) / 100),
            " 0: {:7}, ".format(int(info[0][0][i])),
            " 1: {:7}, ".format(int(info[0][1][i] - info[0][0][i])),
            " 2: {:7}".format(int(info[0][2][i] - info[0][1][i])),
            " sum: {:7}".format(int(info[0][2][i]))
        )
    plt.show()
    return info

def plt_time_alt_tem(df):
    plt.rcParams["figure.figsize"] = [20, 6]
    plt.grid()
    plt.plot(df['date'], df['ALT'])
    plt.plot(df['date'], df['tem'])
    plt.show()
    len_df = len(df['date'])
    tanni = int(len_df/10)
    for i in range(10):
        i = tanni * i
        print("index: ", i)
        plt.plot(pd.to_datetime(df['date']).iloc[i:i+tanni], df['ALT'].iloc[i:i+tanni])
        plt.plot(pd.to_datetime(df['date']).iloc[i:i+tanni], df['tem'].iloc[i:i+tanni])
        plt.ylim([0, 25])
        plt.grid()
        plt.show()