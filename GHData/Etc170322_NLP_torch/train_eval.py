import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import metrics
from tensorboardX import SummaryWriter
import configparser
import time
import os


def init_network(model,method = 'xavier',exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w,0)

def train(model,train_iter,dev_iter,test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = float(model.learning_rate))

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=model.log_path+ '/' + time.strftime('%m%d_%H.%M',time.localtime()))
    for epoch in range(model.epoches):
        model.train()
        print("Epoch [{}/{}]".format(epoch+1,model.epoches))
        for i,(trains,labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs,labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true,predic)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > model.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        writer.close()
        test(model, test_iter)

def test(model, test_iter):
    # test
    model.load_state_dict(torch.load(model.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=model.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

if __name__ == '__main__':
    dataset = 'THUCNews'
    embedding = 'embedding_SougouNews.npz'
    model_class = 'Translate'
    model_name = 'biLSTM_Att'
    exec('from {} import {}'.format(model_class,model_name))
    exec('from {}.utils import build_dataset,build_iterator,get_time_dif'.format(model_class))

    config = configparser.ConfigParser()
    config_path = os.path.join("./config",model_class,model_name+".ini")
    config.read(config_path, encoding='utf-8')
    np.random.seed(1)
    torch.manual_seed(1)

    start_time = time.time()
    print("Loading data......")
    vocab,train_data,dev_data,test_data = build_dataset(config,'True')
    train_iter = build_iterator(train_data,config)
    dev_iter = build_iterator(dev_data,config)
    test_iter = build_iterator(test_data,config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = eval('{}.Model(config)'.format(model_name))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(model, train_iter, dev_iter, test_iter)