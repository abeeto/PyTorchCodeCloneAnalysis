import time
import numpy as np 
import torch 
import torch.utils.data as Data
from torchvision import transforms

from retina_net import RetinaNet 
from dataloader import CocoDataset 
from anchor import Anchor
from focal_loss import FocalLoss 
from utils import collater, Resizer, Augmenter, Normalizer

cuda = torch.cuda.is_available()


def train(batch_size=2, learning_rate=1e-2, train_epoch=100):
    # Normalizer(), Augmenter(), Resizer() 各转换时按顺序进行的
    transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()]) 
    dataset = CocoDataset('./data/coco/', 'train2017', transform)
    data_loader = Data.DataLoader(dataset, 2, num_workers=2, shuffle=True, \
                                  collate_fn=collater, pin_memory=True)
    dataset_size = len(dataset)
    print('sample number:', dataset_size) 
    print('epoch size:', dataset_size / batch_size)

    retinanet = RetinaNet() 
    anchor = Anchor() 
    focal_loss = FocalLoss()

    if cuda:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        anchor = anchor.cuda() 
        focal_loss = focal_loss.cuda()
    retinanet.module.freeze_bn()

    optimizer = torch.optim.SGD(
        retinanet.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    '''
    class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
        factor=0.1, patience=10, verbose=False, threshold=0.0001, 
        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    :param optimer: 指的是网络的优化器
    :param mode: (str), 可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，
                 max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
    :param factor: 学习率每次降低多少，new_lr = old_lr * factor
    :param patience=10: 容忍网路的性能不提升的次数，高于这个次数就降低学习率
    :param verbose: (bool), 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
    :param threshold: (float), 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
    :param cooldown: 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
    :param min_lr: 学习率的下限
    :param eps: 适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
    ————————————————
    版权声明：本文为CSDN博主「张叫张大卫」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/weixin_40100431/article/details/84311430
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    for epoch_num in range(train_epoch):
        epoch_loss = []

        for iter_num, data in enumerate(data_loader):
            iter_time = time.time()
            images, annots, scales = data 
            if cuda:
                images = images.cuda() 
                annots = annots.cuda() 
                scales = scales.cuda() 
            
            total_anchors = anchor(data['img'])
            classification, localization = retinanet(images)

            cls_loss, loc_loss = \
                focal_loss(classification, localization, total_anchors, annots)
            loss = cls_loss + loc_loss
            epoch_loss.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            '''
            关于torch.nn.utils.clip_grad_norm_(): 
            In some cases you may find that each layer of your net amplifies the 
            gradient it receives. This causes a problem because the lower layers of 
            the net then get huge gradients and their updates will be far too large 
            to allow the model to learn anything.

            This function ‘clips’ the norm of the gradients by scaling the gradients 
            down by the same amount in order to reduce the norm to an acceptable 
            level. In practice this places a limit on the size of the parameter 
            updates.

            The hope is that this will ensure that your model gets reasonably 
            sized gradients and that the corresponding updates will allow the 
            model to learn.
            引用自https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873 
            感受一下来自 PyTorch 讨论社区的窒息攻防，2333。。
            '''
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            print(
                '|', 'Epoch:', epoch_num + 1, 
                '|', 'Iter:', iter_num + 1, 
                '|', 'cls loss:', float(cls_loss), 
                '|', 'loc loss:', float(loc_loss), 
                '|', 'loss:', float(loss), 
                '|', 'lr:', float(optimizer.learning_rate), 
                '|', 'time:', time.time() - iter_time
            )
        
        scheduler.step(np.mean(epoch_loss))

        print('Saving parameters in model on epoch', epoch_num + 1)
        torch.save(retinanet.state_dict(), './param/param_epoch'+str(epoch_num+1).zfill(3)+'.pkl')    


if __name__ == '__main__':
    train()