import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from 多个工具 import 获得时间偏移量
from 蟒蛇火炬_预训练.优化器 import 模型的自适应动量估计


def 训练模型(配置, 模型, 训练用迭代器, 验证用迭代器, 测试用迭代器):
    开始时间 = time.time()
    模型.train()
    参数_优化器 = list(模型.named_parameters())
    不_衰减 = ['bias', '层归一化.bias', '层归一化.weight']
    优化器_分组_参数列表 = [
        {'params': [p for n, p in 参数_优化器 if not any(nd in n for nd in 不_衰减)], 'weight_decay': 0.01},
        {'params': [p for n, p in 参数_优化器 if any(nd in n for nd in 不_衰减)], 'weight_decay': 0.0},
    ]
    优化器 = 模型的自适应动量估计(优化器_分组_参数列表, 学习率=配置.学习率, 预热=0.05, 训练_总数=len(训练用迭代器) * 配置.轮回数)

    总计批次 = 0
    验证最佳损失值 = float('inf')
    最后_改善 = 0
    旗帜 = 0
    模型.train()
    for 轮回 in range(配置.轮回数):
        print('轮回 [{}/{}]'.format(轮回 + 1, 配置.轮回数))
        for i, (训练, 标签) in enumerate(训练用迭代器):
            输出 = 模型(训练)
            模型.zero_grad()
            损失值 = F.cross_entropy(输出, 标签)
            损失值.backward()
            优化器.step()
            if 总计批次 % 100 == 0:
                真 = 标签.data.cpu()
                预测 = torch.max(输出.data, 1)[1].cpu()
                训练_准确率 = metrics.accuracy_score(真, 预测)
                验证_准确率, 验证_损失值 = 评估(配置, 模型, 验证用迭代器)
                if 验证_损失值 < 验证最佳损失值:
                    验证最佳损失值 = 验证_损失值
                    torch.save(模型.state_dict(), 配置.保存路径)
                    改善 = '*'
                    最后_改善 = 总计批次
                else:
                    改善 = ''
                时间消耗 = 获得时间偏移量(开始时间)
                消息 = "迭代：{0:>6}，训练模型 损失值：{1:>5.2}，训练模型 准确率：{2:>6.2%}，验证 损失值：{3:>5.2}，验证 准确率：{4:>6.2%}，时间{5} {6}"
                print(消息.format(总计批次, 损失值.item(), 训练_准确率, 验证_损失值, 验证_准确率, 时间消耗, 改善))
                模型.train()
            总计批次 += 1
            if 总计批次 - 最后_改善 > 配置.无效改善阈值:
                print("长时间没有优化，自动停止...")
                旗帜 = True
                break
        if 旗帜:
            break
    测试(配置, 模型, 测试用迭代器)


def 测试(配置, 模型, 测试用迭代器):
    模型.load_state_dict(torch.load(配置.保存路径))
    模型.eval()
    开始时间 = time.time()
    测试_准确率, 测试_损失值, 测试_报告, 测试_混淆 = 评估(配置, 模型, 测试用迭代器, 测试=True)
    消息 = "测试 损失值：{0:>5.2}，测试 准确率：{1:>6.2%}"
    print(消息.format(测试_损失值, 测试_准确率))
    print("精确，召回和F1分数")
    print(测试_报告)
    print("混淆矩阵...")
    print(测试_混淆)
    时间消耗 = 获得时间偏移量(开始时间)
    print("使用时间：", 时间消耗)


def 评估(配置, 模型, 数据_迭代器, 测试=False):
    模型.eval()
    总损失值 = 0
    所有预测 = np.array([], dtype=int)
    所有标签 = np.array([], dtype=int)
    with torch.no_grad():
        for 文本, 标签 in 数据_迭代器:
            输出 = 模型(文本)
            损失值 = F.cross_entropy(输出, 标签)
            总损失值 += 损失值
            标签 = 标签.data.cpu().numpy()
            预测 = torch.max(输出.data, 1)[1].cpu().numpy()
            所有标签 = np.append(所有标签, 标签)
            所有预测 = np.append(所有预测, 预测)

    准确率 = metrics.accuracy_score(所有标签, 所有预测)
    if 测试:
        报告 = metrics.classification_report(所有标签, 所有预测, target_names=配置.类别名单, digits=4)
        混淆 = metrics.confusion_matrix(所有标签, 所有预测)
        return 准确率, 总损失值 / len(数据_迭代器), 报告, 混淆
    return 准确率, 总损失值 / len(数据_迭代器)
