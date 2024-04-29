import platform
import random
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from SSD import SSD as SSD_300
from SSD_NeXt import *
from Loss import MultiBoxLoss, MultiBoxLossNonMatch
from BoundingBox import *
from DataSet import *
from utils.WarmUpLR import WarmUpLR
from Transforms import *
from Argument import *
from Trainer import Trainer

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args = parser_SSD300.parse_args()
args = parser_SSD_NeXt.parse_args()

if platform.system() == 'Windows':
    vocRoot = r'D:\DateSet\VOC'
else:
    vocRoot = r'/root/autodl-tmp/VOC'


def Train():
    """
    训练模型
    """

    '''设置参数'''
    num_epoch = args.num_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    T = num_epoch
    dataset = args.dataset

    '''实例化模型、优化器、损失函数、学习率变化'''
    model = SSD_NeXt(num_classes=20, cfg=SSD_NeXt_cfg).to(device)
    # model = SSD_300(num_classes=20).to(device)

    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': lr}],
                                  lr=lr, weight_decay=weight_decay)

    # criterion = MultiBoxLossNonMatch()
    criterion = MultiBoxLossNonMatch(cfg=model.cfg)

    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    '''加载数据集'''
    # 训练集
    transforms_image = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)],
                               p=0.8),
        RandomNoise(mode=['gaussian', 'localvar', 's&p', 'poisson', 'speckle'], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    transforms_all = RandomChoiceDefinedTrans([
        ComposeDefinedTrans([
            RandomCrop(p=0.8, ratio=(model.h, model.w)),
            LetterBoxResize(size=(model.h, model.w), padding_ratio=0.5, normal_resize=0),
            RandomFlip(lr_ratio=0.5, ud_ratio=0)
        ]),
        Mosaic(
            VOCDetection(root_file=vocRoot, mode='trainval', dataset=dataset,
                         transforms_image=transforms_image, transforms_all=RandomFlip(lr_ratio=0.5, ud_ratio=0)),
            offset=(0.5, 0.5), size=(model.h, model.w), p=1)
    ], p=[5, 5])
    transforms_label = MatchTarget(prior_boxes=model.prior_boxes, cfg=model.cfg, approximate=True)
    # transforms_label = MatchTarget(prior_boxes=model.prior_boxes, approximate=True)
    train_dataset = VOCDetection(root_file=vocRoot, mode='trainval', dataset=dataset, transforms_image=transforms_image,
                                 transforms_all=transforms_all, transforms_label=transforms_label, make_labels=True)
    # 验证集
    transforms_image_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    transforms_all_eval = LetterBoxResize(size=(model.h, model.w))
    eval_dataset = VOCDetection(root_file=vocRoot, mode='test', dataset=dataset,
                                transforms_image=transforms_image_eval, transforms_all=transforms_all_eval)
    # 训练
    trainer = Trainer(args=args,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      train_scheduler=train_scheduler,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      device=device,
                      coco_eval_only=True,
                      wechat_notice=False)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.EvalByCOCO()
    elif args.mode == 'fps':
        trainer.TestFPS()


def TestImage(model, image_file):
    """
    测试检测图片
    :param model: 模型
    :param image_file: 图片文件列表
    """
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    image_transforms = LetterBoxResize(size=(model.h, model.w))
    classes = ['person',
               'car', 'bus', 'bicycle', 'motorbike', 'aeroplane', 'boat', 'train',
               'chair', 'sofa', 'diningtable', 'tvmonitor', 'bottle', 'pottedplant',
               'cat', 'dog', 'cow', 'horse', 'sheep', 'bird']
    for file in image_file:
        image = Image.open(file).convert('RGB')
        w, h = image.size
        input_image = data_transforms(image)
        input_image, _ = image_transforms(input_image, {'anchors': torch.tensor([])})
        objects = model.Predict(input_image.unsqueeze(0).to(device))  # 锚框
        objects = image_transforms.Recover(size=(h, w), anchors=objects)
        original_image = torchvision.io.read_image(file).permute(1, 2, 0)
        display(original_image, objects.cpu(), classes=classes)


def CompareModel():
    """
    比较SSD-NeXt和SSD300对KITTI验证集图片的检测效果
    """
    SSD_next = SSD_NeXt(num_classes=2, cfg=SSD_NeXt_cfg).to(device)
    SSD300 = SSD_300(num_classes=2).to(device)

    SSD_next.load_state_dict(torch.load('weights/model_SSD-NeXt_KITTI.pth'))
    SSD300.load_state_dict(torch.load('weights/SSD300_KITTI.pth')['model'])

    transforms_image_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    transforms_all_eval = LetterBoxResize(size=(SSD_next.h, SSD_next.w))
    eval_dataset = VOCDetection(root_file=vocRoot, mode='val', dataset='KITTI',
                                transforms_image=transforms.ToTensor())
    eval_dataset1 = VOCDetection(root_file=vocRoot, mode='val', dataset='KITTI',
                                 transforms_image=transforms_image_eval, transforms_all=transforms_all_eval)
    eval_dataset2 = VOCDetection(root_file=vocRoot, mode='val', dataset='KITTI',
                                 transforms_image=transforms.ToTensor(),
                                 transforms_all=LetterBoxResize(size=(300, 300)))
    eval_dataset3 = VOCDetection(root_file=vocRoot, mode='val', dataset='KITTI',
                                 transforms_image=transforms.ToTensor(), transforms_all=LetterBoxResize(size=(1200, 1200)))

    sequence = list(range(len(eval_dataset1)))
    random.shuffle(sequence)
    for i in tqdm(sequence, desc='\teval'):
        img, _ = eval_dataset[i]
        img1, _ = eval_dataset1[i]
        img2, _ = eval_dataset2[i]
        img3, ob3 = eval_dataset3[i]

        anchors = ob3['anchors']
        anchors[:, -1] -= 1
        mask = anchors[:, -1] < 2
        anchors = anchors[mask]
        anchors = torch.cat((anchors, torch.ones((anchors.shape[0], 1))), dim=1)
        plt.xticks(alpha=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.tick_params(axis='y', width=0)
        display(img3.permute(1, 2, 0), anchors, 0.5, classes=eval_dataset1.classes, show_score=False)

        # 在测试阶段，使用no_grad()语句停止求梯度，节省算力和显存，否则容易显存溢出
        with torch.no_grad():
            object_preds1 = SSD_next.Predict(img1.unsqueeze(0).to(device), IOU_threshold=0.4, conf_threshold=0.4)[0]
            # object_preds2 = SSD300.Predict(img2.unsqueeze(0).to(device), IOU_threshold=0.4, conf_threshold=0.4)[0]
        h, w = eval_dataset1.GetHW(index=i)
        object_preds1 = transforms_all_eval.Recover(size=(h, w), anchors=object_preds1)
        object_preds2 = transforms_all_eval.Recover(size=(h, w), anchors=object_preds2)
        plt.xticks(alpha=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.tick_params(axis='y', width=0)
        display(img.permute(1, 2, 0), object_preds1.cpu(), 0.5, classes=eval_dataset1.classes, show_score=False)
        plt.xticks(alpha=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.tick_params(axis='y', width=0)
        display(img.permute(1, 2, 0), object_preds2.cpu(), 0.5, classes=eval_dataset1.classes, show_score=False)
        time.sleep(1)


def TestForTrain():
    """
    使用少量几张图片训练模型，以验证是否能够正常收敛
    """
    batch_size = 3
    warm_up = 5
    model = SSD_NeXt(num_classes=20, cfg=SSD_NeXt_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = MultiBoxLossNonMatch(cfg=model.cfg)

    warmup_scheduler = WarmUpLR(optimizer, warm_up)

    # 加载数据集
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    image_transforms = ComposeDefinedTrans(
        [LetterBoxResize(size=(model.h, model.w), padding_ratio=0.5, normal_resize=0)])
    # label_transforms = MatchTarget(prior_boxes=model.prior_boxes, approximate=True)
    label_transforms = MatchTarget(prior_boxes=model.prior_boxes, cfg=model.cfg, approximate=True)
    print(model.prior_boxes.shape)
    train = VOCDetection(root_file=vocRoot, mode='trainval', dataset=args.dataset,
                         transforms_image=data_transforms, transforms_all=image_transforms,
                         transforms_label=label_transforms, make_labels=True)
    voc_iter = DataLoader(train, batch_size=batch_size, collate_fn=train.collate_matched, shuffle=True)
    classes = train.classes
    for X, Y in voc_iter:
        if Y['anchors'][2].sum() / 4 < 5 * batch_size:
            continue
        print(Y['anchors'][2].sum() / 4)
        img_ids = Y['image_ids']
        print(img_ids)
        model.train()
        for epoch in range(100):
            images = X.to(device)
            targets = Y['anchors'][0].to(device), Y['anchors'][1].to(device), Y['anchors'][2].to(device)
            predictions = model(images)
            optimizer.zero_grad()

            if epoch == 90:
                a = 1
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if epoch < warm_up:
                warmup_scheduler.step()

            if (epoch + 1) % 10 == 0:
                print('loss:', loss.item())

        images = X.to(device)
        with torch.no_grad():
            batch_output = model.Predict(images)

        for batch in range(batch_size):
            # 读取图片
            img = X[batch].permute(1, 2, 0)
            output = batch_output[batch]
            print(output, output.shape)
            display(img, output.cpu(), classes=classes)
            train.ShowImage(img_ids[batch])
        break


if __name__ == "__main__":
    print(device)
    print(args)
    Train()
    # TestOneImage()
    # CompareModel()
