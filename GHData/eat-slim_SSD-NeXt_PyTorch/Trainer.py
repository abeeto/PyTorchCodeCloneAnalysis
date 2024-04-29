import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from pycocotools.cocoeval import COCOeval
from utils.Timer import *
from utils.WarmUpLR import WarmUpLR
from utils.summarize import summarize
from Transforms import *
import platform
import requests


class Trainer:
    """
    封装训练与测试过程
    """
    def __init__(self, args, model, optimizer, criterion, train_scheduler, train_dataset, eval_dataset, device,
                 coco_eval_only=False, wechat_notice=False):
        self.args = args
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_scheduler = train_scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        self.coco_eval_only = coco_eval_only
        self.wechat_notice = wechat_notice

        self.name = '_'.join([args.name, args.dataset, args.version, str(args.lr), str(args.weight_decay)])
        self.start_epoch = 0
        self.step = 0
        if self.args.checkpoint != '':
            checkpoint = torch.load(os.path.join(self.args.save_folder, self.args.checkpoint))
            self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_scheduler' in checkpoint:
                self.train_scheduler.load_state_dict(checkpoint['train_scheduler'])
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1
            if 'name' in checkpoint:
                self.name = checkpoint['name']
            print('loading checkpoint complete')

        if platform.system() == 'Windows':
            self.vocRoot = r'D:\DateSet\VOC'
            self.log = f'log/{self.name}'
        else:
            self.vocRoot = r'/root/autodl-tmp/VOC'
            self.log = f'/root/tf-logs/{self.name}'

        if self.args.set_lr > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.set_lr

        self.classes = self.eval_dataset.classes
        if self.eval_dataset.dataset == 'KITTI':
            annFile = os.path.join(self.vocRoot, 'KITTI', 'val.json')
            self.dont_care = len(self.classes)
            self.classes = self.classes[:-1]  # 最后一个标签为dont care，忽略即可
        else:
            annFile = os.path.join(self.vocRoot, 'VOCtest', 'test2007.json')
            self.dont_care = 0
        self.coco = COCO(annFile)

    def train(self):
        """训练"""
        if self.start_epoch < self.args.unfreeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        if self.start_epoch < self.args.warm_up:
            warmup_scheduler = WarmUpLR(self.optimizer, self.args.warm_up, last_epoch=self.start_epoch)

        voc_iter = DataLoader(self.train_dataset,
                              batch_size=self.args.batch_size,
                              collate_fn=self.train_dataset.collate_matched,
                              num_workers=self.args.num_workers,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True)

        '''记录训练过程数据'''
        # tensorboard可视化训练过程，记录训练时的相关数据，使用指令:tensorboard --logdir=log
        writer = SummaryWriter(self.log)
        init_image = torch.rand((1, 3, self.model.h, self.model.w), dtype=torch.float32, device=self.device)
        writer.add_graph(self.model, init_image)

        # 其他数据
        train_timer = Timer()
        eval_timer = Timer()
        step_timer = Timer()
        # 根据每个step包含的样本数量计算出每个step对应的batch数量
        batch_per_step = int(self.args.sample_per_step / self.args.batch_size)
        count = batch_per_step
        batch_loss = []  # 记录每个batch的loss

        for epoch in range(self.args.num_epoch)[self.start_epoch:]:
            print('epoch:', epoch + 1)
            self.model.train()
            batch_loss.clear()
            if self.args.unfreeze == epoch and self.args.unfreeze != 0:
                for p in self.model.backbone.parameters():
                    p.requires_grad = True

            # 一轮训练开始
            print("学习率：%f" % (self.optimizer.param_groups[0]['lr']))
            train_timer.start()
            step_timer.start()
            for X, Y in voc_iter:
                images = X.to(self.device)
                targets = Y['anchors'][0].to(self.device), \
                          Y['anchors'][1].to(self.device), \
                          Y['anchors'][2].to(self.device)
                self.optimizer.zero_grad()

                # with autocast():
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)

                loss.backward()
                self.optimizer.step()

                # 记录训练数据
                batch_loss.append(loss.item())
                count -= 1
                if count == 0:
                    count = batch_per_step
                    print('\tloss =', loss.data, '\t{:.2f} sec'.format(step_timer.stop()))
                    step_timer.start()
                    writer.add_scalar("train_loss", loss, self.step)
                    self.step += 1

            if epoch + 1 < self.args.warm_up:
                warmup_scheduler.step()
            else:
                self.train_scheduler.step()
            train_time = train_timer.stop()

            # 保存参数
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                     'train_scheduler': self.train_scheduler.state_dict(), 'epoch': epoch, 'step': self.step,
                     'name': self.name}
            torch.save(state, os.path.join(self.args.save_folder, f"{self.name}_epoch{epoch + 1}.pth"))

            # 记录训练效果
            mean_loss = sum(batch_loss) / len(batch_loss)
            writer.add_scalar("loss_mAP/loss", mean_loss, epoch)
            print('\tmean loss = {:.6f}'.format(mean_loss))
            print('\ttrain time = {:.2f}sec'.format(train_time))
            # 每隔几轮训练后使用验证集验证
            # if (epoch + 1) % self.args.eval_frequency == 0 and epoch + 1 >= self.args.warm_up:
            if (epoch + 1) % self.args.eval_frequency == 0:
                eval_timer.start()
                if self.coco_eval_only:
                    mAP, AP_dict = self.EvalByCOCO()
                else:
                    mAP, AP_dict = self.eval()
                eval_time = eval_timer.stop()

                writer.add_scalar("loss_mAP/mAP", mAP, epoch)
                print('\tmAP ={:.4f}\t'.format(mAP), 'AP_dict =', AP_dict)
                print('\teval time = {:.2f}sec'.format(eval_time))
                if self.wechat_notice:
                    self.WechatNotice(epoch, '{:.6f}'.format(mean_loss), '{:.4f}'.format(mAP))

        torch.save(self.model.state_dict(), os.path.join(self.args.save_folder, f"{self.name}.pth"))
        print('\tmean train time =', train_timer.avg(), 'sec')

    def eval(self):
        """验证"""
        detection_tables = []  # 保存每个类别在验证集上的检测结果
        num_gt = [0] * len(self.classes)  # 保存每个类别真实框的总数
        is_exist = [False] * len(self.classes)  # 记录每个类别是否存在
        resize = LetterBoxResize(size=(self.model.h, self.model.w))
        result = []  # 保存检测结果
        for cls in range(len(self.classes)):
            # 每个类别用一张表格记录，每行代表一个该类别预测框的预测结果，每列分别表示[置信度, 是否为TP, 是否为FP]
            detection_tables.append(torch.tensor([], dtype=torch.float32, device=self.device))

        # 对验证集中所有图片进行检测
        eval_iter = DataLoader(self.eval_dataset, batch_size=self.args.test_bsz, shuffle=False,
                               num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_fn,
                               pin_memory=True)

        for X, Y in tqdm(eval_iter, desc='\teval'):
            img, target = X.to(self.device), Y['anchors'][0].to(self.device)  # 真实结果[真实框数量, 5]

            # 在测试阶段，使用no_grad()语句停止求梯度，节省算力和显存，否则容易显存溢出
            with torch.no_grad():
                batch_object_preds = self.model.Predict(img)  # 获取图片预测到的最终结果[batch_size, 预测框数量, 6]

            for batch in range(len(batch_object_preds)):
                object_preds = batch_object_preds[batch]

                if object_preds.shape[0] == 0:
                    continue

                img_id = Y['image_ids'][batch]
                object_preds = resize.Recover(self.eval_dataset.GetHW(img_id=img_id), object_preds)
                target = resize.Recover(self.eval_dataset.GetHW(img_id=img_id), target)

                # 去除dont care区域的所有预测
                if self.dont_care > 0:
                    dont_care_mask = target[:, -1] == self.dont_care
                    dont_care_area = target[dont_care_mask][:, :-1]  # 区域范围
                    if object_preds.shape[0] > 0:
                        center = (object_preds[:, [2, 3]] + object_preds[:, [0, 1]]) / 2  # 所有预测框的中心
                        mask = torch.full((object_preds.shape[0],), False, dtype=torch.bool, device=object_preds.device)
                        for area in dont_care_area:
                            # 若预测框中心在区域内，则置为True
                            mask += (center[:, 0] > area[0]) & (center[:, 0] < area[2]) & \
                                    (center[:, 1] > area[1]) & (center[:, 1] < area[3])
                        object_preds = object_preds[mask == False]  # 取反后，区域内的预测框被置为False而舍弃
                    target = target[dont_care_mask == False]

                # coco API格式
                coco_preds = object_preds.clone()
                for anchor in coco_preds:
                    h, w = self.eval_dataset.GetHW(img_id=img_id)
                    anchor[:4] *= torch.tensor([w, h, w, h], device=anchor.device)  # 转化为在原图上的对角坐标
                    anchor[2:4] -= anchor[:2]  # 转换为[xmin, ymin, w, h]的格式
                    result.append({
                        "image_id": int(img_id),
                        "category_id": int(anchor[-2] + 1),
                        "bbox": anchor[:4].tolist(),
                        "score": anchor[-1].item()
                    })

                target[:, -1] -= 1  # 原始数据集中的图片标签是+1的，需要减掉
                for cls in range(len(self.classes)):
                    # 遍历所有类别，计算每个类别预测框的匹配情况
                    class_target = target[target[:, -1] == cls]
                    num_gt[cls] += class_target.shape[0]
                    if class_target.shape[0] != 0:
                        is_exist[cls] = True

                    class_preds = object_preds[object_preds[:, -2] == cls]

                    if class_preds.shape[0] != 0:
                        is_exist[cls] = True
                        iou = IOU(class_preds[:, :4], class_target[:, :4])  # 计算iou，返回[预测框数量, 真实框数量]
                        TP = torch.full((class_preds.shape[0],), False, dtype=torch.bool,
                                        device=self.device)  # 保存和真实框相匹配的TP

                        # 依次给每个真实框寻找最大iou的预测框，保证所有真实框匹配的预测框不重复
                        num_target = class_target.shape[0]  # 获取真实框数量

                        for _ in range(num_target):
                            if torch.max(iou) < 0.5:
                                break
                            max_idx = torch.argmax(iou)  # 找到矩阵中最大值，得到其索引，该索引为一维索引
                            target_idx = (max_idx % num_target).long()  # 将索引转化为行列形式索引，找到对应的预设框与真实框的索引
                            preds_idx = (max_idx / num_target).long()
                            TP[preds_idx] = True  # 找到一个正例
                            iou[:, target_idx] = -1  # 使用-1填充，代表剔除该行和列
                            iou[preds_idx, :] = -1

                        FP = (TP == False)
                        table = torch.zeros((class_preds.shape[0], 3), dtype=torch.float32, device=self.device)
                        table[:, 0] = class_preds[:, -1]  # 填充表格内容
                        table[:, 1][TP] = 1
                        table[:, 2][FP] = 1

                        detection_tables[cls] = torch.cat((detection_tables[cls], table), dim=0)  # 拼接到总体表格中

        # json格式保存检测结果，交给coco API评估
        if len(result) == 0:
            result.append({
                "image_id": int(img_id),
                "category_id": 0,
                "bbox": [0, 0, 0, 0],
                "score": 0
            })
        json_str = json.dumps(result, indent=4)
        with open('predict_results.json', 'w') as json_file:
            json_file.write(json_str)

        coco_pre = self.coco.loadRes('predict_results.json')
        coco_evaluator = COCOeval(cocoGt=self.coco, cocoDt=coco_pre, iouType="bbox")
        print('-----------------------------------------')
        coco_evaluator.evaluate()
        print('-----------------------------------------')
        coco_evaluator.accumulate()
        print('-----------------------------------------')
        coco_evaluator.summarize()

        AP = []
        for i in tqdm(range(len(self.classes)), desc='\tcomputing AP'):
            AP.append(self.ComputeAP(detection_tables[i], num_gt[i]))
        mAP = sum(AP) / sum(is_exist)
        AP_dict = {k: v for k, v in zip(self.classes, AP)}
        return float(mAP), AP_dict

    @staticmethod
    def ComputeAP(detection_table, num_gt, before_VOC2010=False):
        """计算AP"""
        if detection_table.shape[0] == 0 or num_gt == 0:
            return 0
        _, indices = torch.sort(detection_table[:, 0], descending=True)  # 根据置信度排序，获得排序索引
        detection_table = detection_table[indices]  # 重新调整顺序，按置信度由高到低
        accTP = torch.cumsum(detection_table[:, 1], dim=0)
        accFP = torch.cumsum(detection_table[:, 2], dim=0)
        precision = accTP / (accTP + accFP)  # 准确度
        recall = accTP / num_gt  # 召回率

        _, recall_sort_indices = torch.sort(recall)  # 升序排列recall
        recall = recall[recall_sort_indices]  # 根据升序索引调整顺序
        precision = precision[recall_sort_indices]
        AP = 0
        # 根据P-R曲线计算AP
        if not before_VOC2010:
            last_recall = 0
            # 从最小的recall开始计算
            for i in range(len(recall)):
                # 每个不同的recall计算一次
                if recall[i] == last_recall:
                    continue
                max_precision = torch.max(precision[i:])  # 找到当前位置到最后的最大的precision
                AP += max_precision * (recall[i] - last_recall)  # precision与两个相邻recall差值的乘积
                last_recall = recall[i]

        # VOC2010之前采取另一种计算方法,分别选取0-1之间11处recall对应的最大precision，再求和取平均
        else:
            for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                mask = recall > i
                try:
                    idx = mask.tolist().index(True)  # 找到该recall区间的起始位置
                except:
                    break
                AP += torch.max(precision[idx:])  # 找到每个区间起始位置到最后的最大的precision并求和
            AP /= 11  # 取平均
        return float(AP)

    def EvalByCOCO(self):
        """仅使用COCO的API进行验证"""
        resize = LetterBoxResize(size=(self.model.h, self.model.w))
        result = []  # 保存检测结果
        # 对验证集中所有图片进行检测
        eval_iter = DataLoader(self.eval_dataset, batch_size=self.args.test_bsz, shuffle=False,
                               num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_fn,
                               pin_memory=True)

        for X, Y in tqdm(eval_iter, desc='\teval'):
            img, batch_target = X.to(self.device), Y['anchors'].to(self.device)  # 真实结果[batch_size, 真实框数量, 5]

            # 在测试阶段，使用no_grad()语句停止求梯度，节省算力和显存，否则容易显存溢出
            with torch.no_grad():
                batch_object_preds = self.model.Predict(img)  # 获取一张图片预测到的最终结果[预测框数量, 6]

            for batch in range(len(batch_object_preds)):
                object_preds = batch_object_preds[batch]

                if object_preds.shape[0] == 0:
                    continue

                # 去除dont care区域的所有预测
                if self.dont_care > 0:
                    target = batch_target[batch]
                    target = target[target[:, -1] > 0]
                    dont_care_mask = target[:, -1] == self.dont_care
                    dont_care_area = target[dont_care_mask][:, :-1]  # 区域范围
                    if object_preds.shape[0] > 0:
                        center = (object_preds[:, [2, 3]] + object_preds[:, [0, 1]]) / 2  # 所有预测框的中心
                        mask = torch.full((object_preds.shape[0],), False, dtype=torch.bool, device=object_preds.device)
                        for area in dont_care_area:
                            # 若预测框中心在区域内，则置为True
                            mask += (center[:, 0] > area[0]) & (center[:, 0] < area[2]) & \
                                    (center[:, 1] > area[1]) & (center[:, 1] < area[3])
                        object_preds = object_preds[mask == False]  # 取反后，区域内的预测框被置为False而舍弃

                # coco API格式
                img_id = Y['image_ids'][batch]
                h, w = self.eval_dataset.GetHW(img_id=img_id)
                object_preds = resize.Recover(size=(h, w), anchors=object_preds)
                object_preds[:, :4] *= torch.tensor([w, h, w, h], device=self.device)  # 转化为在原图上的对角坐标
                object_preds[:, 2:4] -= object_preds[:, :2]  # 转换为[xmin, ymin, w, h]的格式
                for anchor in object_preds:
                    result.append({
                        "image_id": int(img_id),
                        "category_id": int(anchor[-2] + 1),
                        "bbox": anchor[:4].tolist(),
                        "score": anchor[-1].item()
                    })

        # json格式保存检测结果，交给coco API评估
        if len(result) == 0:
            result.append({
                "image_id": int(img_id),
                "category_id": 0,
                "bbox": [0, 0, 0, 0],
                "score": 0
            })
        json_str = json.dumps(result, indent=4)
        with open('predict_results.json', 'w') as json_file:
            json_file.write(json_str)

        coco_pre = self.coco.loadRes('predict_results.json')
        coco_evaluator = COCOeval(cocoGt=self.coco, cocoDt=coco_pre, iouType="bbox")
        coco_evaluator.params.maxDets = [1, 20, 200]
        print('-----------------------------------------')
        coco_evaluator.evaluate()
        print('-----------------------------------------')
        coco_evaluator.accumulate()
        print('-----------------------------------------')
        coco_evaluator.summarize()

        # 计算每一类的AP(IoU=0.5)
        AP_dict = {}
        for i in range(len(self.classes)):
            stats, _ = summarize(coco_evaluator, catId=i)
            AP_dict[self.classes[i]] = stats[1]

        print('-----------------------------------------')
        for key, value in AP_dict.items():
            print(" {:15}: {:.3f}".format(key, value))

        mAP = coco_evaluator.stats[1]
        return float(mAP), AP_dict

    def TestFPS(self):
        """测试FPS"""
        eval_iter = DataLoader(self.eval_dataset, batch_size=self.args.test_bsz, shuffle=False,
                               num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_fn,
                               pin_memory=True)
        timer = Timer()
        for X, _ in tqdm(eval_iter, desc='\teval'):
            img = X.to(self.device)
            # 在测试阶段，使用no_grad()语句停止求梯度，节省算力和显存，否则容易显存溢出
            timer.start()
            with torch.no_grad():
                self.model.Predict(img, conf_threshold=0.5)  # 获取一张图片预测到的最终结果[预测框数量, 6]
            timer.stop()
        infer_time = timer.sum() / len(self.eval_dataset)
        fps = 1 / infer_time
        print('batch size:', self.args.test_bsz)
        print('infer time: {:.2f} ms'.format(infer_time * 1000))
        print('FPS: {:.2f}'.format(fps))
        return infer_time * 1000, fps

    def WechatNotice(self, epoch, loss, mAP):
        """微信提醒训练结果"""
        resp = requests.post(
            "https://xxx/wechat/xxx",  # 替换成你的微信消息通知地址
            json={
                "token": "xxx",  # 替换成你的token
                "title": "来自炼丹炉",
                "name": self.name,
                "content": f"epoch={epoch}\nloss={loss}\nmAP={float(mAP) * 100}%"
            }
        )
        print(resp.content.decode())
