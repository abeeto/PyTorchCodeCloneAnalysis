import os
import torch
from torch import optim
from torch import nn
from VGG import VGG16, load_data, train_model, test_model, cost_visualization, acc_visualization

if __name__ == '__main__':
    # 设置超参数
    batch_size, lr, epoches = 256, 0.03, 200

    # 是否需要训练模型，训练好之后改为False
    train_flag = False

    # 加载数据
    train_dataset, train_loader, test_dataset, test_loader = load_data(batch_size)

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 列表记录损失与准确率用于可视化
    LOSS, ACC = [], []

    # 创建模型
    model = VGG16()
    model.parameters()

    # 使用GPU
    gpu_is_available = torch.cuda.is_available()
    if gpu_is_available:
        model.cuda()

    # 优化
    # 这里尝试了各种方法
    # 最终选择了：带有动量的SGD
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 100, 160], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    while 1:
        if train_flag:
            # 训练模型
            for epoch in range(epoches):
                L = train_model(model, train_loader,
                                optimizer, criterion,
                                scheduler, train_dataset,
                                epoch, gpu_is_available)

                acc = test_model(model, test_loader, test_dataset, gpu_is_available)

                # 损失与准确率加入列表
                LOSS.append(L)
                ACC.append(acc)

            # 损失函数与准确率的可视化
            cost_visualization(epoches, LOSS)
            acc_visualization(epoches, ACC)

            # 保存模型
            torch.save(model.state_dict(), "./VGG16.pth")
            exit(0)
        elif os.path.exists("./VGG16.pth") and train_flag == False:
            # 测试保存的模型在测试集上的准确率
            model.load_state_dict(torch.load("./VGG16.pth"))
            test_model(model, test_loader, test_dataset, gpu_is_available)
            print("模型参数: ", model)
            exit(0)
        else:
            train_flag = True
