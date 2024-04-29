import sys
import os
import json
from optparse import OptionParser
from PIL import Image
import numpy as np

import torch
import torch.nn as nn

from algorithm import setupAlgorithm
from dataset import setupDataset
from model import setupModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readCommand(argv):
    usageStr = """
    USAGE:      python run_attack.py <options>
    EXAMPLES:   (1) python run_attack.py
                      - 以默认参数开始攻击（使用FGSM在cifar-10数据集上对预训练的resnet-20网络进行攻击）
                (2) python run_attack.py -a PGD -d cifar10 -m resnet110
                      - 使用PGD在cifar-10数据集上对预训练的resnet110网络进行攻击
    """
    parser = OptionParser(usageStr)

    parser.add_option('-a', '--algorithm', help='Choose an Attack Algorithm',
                      type="string", default='FGSM')
    parser.add_option('-d', '--dataset', help='Choose a Dataset',
                      type="string", default='cifar10')
    parser.add_option('-m', '--model', help='Choose a Model to Attack',
                      type="string", default='resnet20')

    options, _ = parser.parse_args(argv)
    args = dict()
    args['chosenAlgorithm'] = options.algorithm
    args['chosenDataset'] = options.dataset
    args['chosenModel'] = options.model

    return args


def setupAttack(chosenAlgorithm, chosenDataset, chosenModel):

    algorithm = setupAlgorithm(chosenAlgorithm)
    dataloader, classes, mean, std = setupDataset(chosenDataset)
    model = setupModel(chosenModel)
    return algorithm, dataloader, classes, model, mean, std


def runAttack(algorithm, dataloader, mean, std, model, loss_fn):
    model.eval()
    acc_counter = 0
    adv_examples = []
    attack = algorithm(model, loss_fn, std)

    # 循环测试集中的所有样本
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        perturbed_data = attack.perturb(data, target)
        model.zero_grad()
        perturbed_output = model(perturbed_data)
        perturbed_pred = perturbed_output.argmax(dim=1)
        acc_counter += (perturbed_pred == target).sum().item()

        adv_ex = ((perturbed_data.cpu()) * std + mean).clamp(0, 1)
        adv_ex = (adv_ex * 255).clamp(0, 255)
        adv_ex = adv_ex.detach().data.numpy().round()
        adv_ex = adv_ex.transpose((0, 2, 3, 1))

        target = target.cpu()
        perturbed_pred = perturbed_pred.cpu()
        for i in range(len(target)):
            adv_examples.append((target[i], perturbed_pred[i], adv_ex[i]))

    final_acc = acc_counter / float(len(dataloader.dataset))
    print('Accuracy = {} / {} = {}(可能因为将最后一个不满的batch丢弃导致实际总数与显示不同)'
          .format(acc_counter, len(dataloader.dataset), final_acc))

    return adv_examples


def saveAdvExamples(adv_examples, classes, chosenAlgorithm, chosenDataset, chosenModel):

    try:
        os.mkdir('./output')
    except:
        pass
    folder_name = './output/' + str(chosenAlgorithm) + '-' + str(chosenDataset) + '-' + str(chosenModel)
    try:
        folder_name_final = folder_name + '/'
        os.mkdir(folder_name_final)
    except:
        except_count = 1
        while True:
            try:
                folder_name_final = folder_name + '-0' + str(except_count) + '/'
                os.mkdir(folder_name_final)
                break
            except:
                except_count += 1
                pass

    adv_count = 0
    print('生成的对抗样本将存储在 {} 文件夹下！'.format(folder_name_final))
    for adv_ex in adv_examples:
        pred = adv_ex[0]
        perturbed_pred = adv_ex[1]
        adv_example = adv_ex[2]
        adv_example = Image.fromarray(adv_example.astype(np.uint8))
        name = folder_name_final + str(adv_count) + '-' + str(classes[pred]) + '-' + str(classes[perturbed_pred]) + '.jpg'
        adv_example.save(name)
        adv_count += 1


if __name__ == '__main__':

    # options 为一个dict字典
    options = readCommand(sys.argv[1:])
    algorithm, dataloader, classes, model, mean, std = setupAttack(**options)

    model = model.to(device)

    with open('config.json') as config_file:
        config = json.load(config_file)
    loss_fn_choice = config['loss_fn']
    if loss_fn_choice == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise Exception("其他损失函数暂不支持! 请使用README中所述的已支持损失函数！")

    adv_examples = runAttack(algorithm, dataloader, mean, std, model, loss_fn)
    saveAdvExamples(adv_examples, classes, **options)
