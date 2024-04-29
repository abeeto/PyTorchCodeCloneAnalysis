import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MyData_kfold import train_dataloader,val_dataloader
#from deeplabv3 import resnet50, resnet101,resnet152, ResNet
#from deeplabv3p.deeplabv3_plus import DeepLab
#from CCNet.ccnet import resnet152
#from deeplabv3_dan_0408 import resnet152
#from dan_v3_v0420 import resnet152
from ccnet_v3_v0509 import resnet152
from tensorboardX import SummaryWriter
from MIouv0217 import Evaluator



def train(epoch = 400):
    # 创建指标计算对象
    evaluator = Evaluator(4)

    # 定义最好指标miou数值，初始化为0
    best_pred = 0.0
    writer = SummaryWriter('tblog/ccnet_v3_0509')
    # 指定第二块gpu
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 模型建立
    #deeplabv3_model = resnet152()
    #deeplabv3plus_model = DeepLab(backbone='resnet', output_stride=16)
    #deeplabv3_model = torch.load('checkpoints/deeplabv3_model_90.pt')
    #ccnet_model = resnet152()
    ccnet_v3_model = resnet152()
    ccnet_v3_model = ccnet_v3_model.to(device)
    #ccnet_model = ccnet_model.to(device)
    #deeplabv3_model = deeplabv3_model.to(device)
    #deeplabv3plus_model = deeplabv3plus_model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device) # CrossEntropyLoss适用多分类
    optimizer = optim.Adam(ccnet_v3_model.parameters(), lr=1e-3)

    for epo in range(epoch):
        # 每个epoch都要记录5次交叉验证的train_loss和val_loss，最后除5
        train_loss = 0
        val_loss = 0
        val_acc = 0
        val_miou = 0
        for i in range(5):
            # 训练部分
            ccnet_v3_model.train()
            #deeplabv3_model.train()
            for index, (image, label) in enumerate(train_dataloader[i]):
                image = image.to(device) 
                label = label.to(device)
                #output = deeplabv3_model(image)
                #output = deeplabv3plus_model(image)
                output = ccnet_v3_model(image)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                iter_loss = loss.item() # 取出数值
                train_loss += iter_loss
                optimizer.step()

                if np.mod(index, 24) == 0:
                    line = "epoch {}_{}, {}/{},train loss is {}".format(epo, i, index, len(train_dataloader[i]), iter_loss)
                    print(line)
                    # 写到日志文件
                    with open('log/logs_ccnet_v3_0509.txt', 'a') as f :
                        f.write(line)
                        f.write('\r\n')

            # 验证部分
            ccnet_v3_model.eval()
            with torch.no_grad():
                for index, (image, label) in enumerate(val_dataloader[i]):
                    image = image.to(device)
                    label = label.to(device)
                    
                    #output = deeplabv3_model(image)
                    #output = deeplabv3plus_model(image)
                    output = ccnet_v3_model(image)
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    iter_loss = loss.item()
                    val_loss += iter_loss
                    # 记录相关指标数据
                    pred = output.cpu().numpy()
                    label = label.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    evaluator.add_batch(label, pred)
                Acc = evaluator.Pixel_Accuracy()
                mIoU = evaluator.Mean_Intersection_over_Union()
                val_acc += Acc
                val_miou += mIoU
                evaluator.reset() # 该5次求指标，每次求之前先清零
        line_epoch = "epoch train loss = %.3f, epoch val loss = %.3f" % (train_loss/len(train_dataloader[i])/5, val_loss/len(val_dataloader[i])/5)
        print(line_epoch)
        with open('log/logs_ccnet_v3_0509.txt', 'a') as f :
            f.write(line_epoch)
            f.write('\r\n')
        
        #Acc = evaluator.Pixel_Accuracy()
        #Acc_class = evaluator.Pixel_Accuracy_Class()
        #mIoU = evaluator.Mean_Intersection_over_Union()
        # tensorboard记录
        writer.add_scalar('train_loss', train_loss/len(train_dataloader[i])/5, epo)
        writer.add_scalar('val_loss', val_loss/len(val_dataloader[i])/5, epo)
        writer.add_scalar('val_Acc', val_acc/5, epo)
        #writer.add_scalar('Acc_class', Acc_class, epo)
        writer.add_scalar('val_mIoU', val_miou/5, epo)        
        
        # 每次验证，根据新得出的miou指标来保存模型
        #global best_pred
        new_pred = val_miou/5
        if new_pred > best_pred:
            best_pred = new_pred
            torch.save(ccnet_v3_model.state_dict(), 'models_ccnet_v3_0509/ccnet_v3_{}.pth'.format(epo))
            #torch.save(deeplabv3_model.state_dict(), 'models_v0304_pre/deeplabv3_{}.pth'.format(epo))
            #torch.save(deeplabv3plus_model, 'checkpoints_v3p_v0316/deeplabv3plus_model_{}.pt'.format(epo))
        '''
        # 每5轮保存一下模型
        if np.mod(epo, 5) == 0:
            torch.save(deeplabv3_model, 'checkpoints_v0225/deeplabv3_model_{}.pt'.format(epo))
            print('saving checkpoints_v0225/deeplabv3_model_{}.pt'.format(epo))
        '''

if __name__ == "__main__":
    train(epoch=40)
