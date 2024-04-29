import os
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
import fire
import time

import models   # 模型包
from data.dataset import DogCat # 数据集
from config import opt  # 参数配置（对象）
from utils import Visualizer # 可视化（visdom）


'''Test 与 保存'''
def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


'''Train 与 Validation'''
def val(model, dataloader):
    model.eval() # 模型设为验证模式

    confusion_matrix = meter.ConfusionMeter(k=2)
    for ii,(input, label) in enumerate(dataloader):
        with t.no_grad():
            val_input = input
            val_label = label.long()
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        with t.no_grad(): # 加上这行，否则在运行ResNet34时会报错“CUDA out of memory”
            score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), val_label)

    model.train() # 切换回训练模式
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def train(**kwargs):
    opt._parse(kwargs) # 根据命令行参数更新配置
    vis = Visualizer(opt.env)

    # （1）模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:                 # load
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()            # cuda

    # （2）数据
    train_data = DogCat(opt.train_data_root, train=True, test=False)
    val_data = DogCat(opt.train_data_root, train=False, test=False)
    train_dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # （3）目标函数 & 优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    # optimizer = t.optim.Adam(params=model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    optimizer = t.optim.SGD(params=model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    #（4）统计指标：平滑处理后的损失 & 混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(k=2)
    previous_loss = 1e100

    #（5）训练
    for epoch in range(opt.max_epoch):
        print("epoch " + str(epoch) + " " + time.strftime('%y%m%d_%H:%M:%S'))
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in enumerate(train_dataloader):
            # 训练模型参数
            input = data
            target = label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)
            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss', loss_meter.value()[0])
                if os.path.exists(opt.debug_file): # 如果需要的话，进入debug模式
                    import ipdb
                    ipdb.set_trace()
        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0]>previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__ == "__main__":
    fire.Fire()


