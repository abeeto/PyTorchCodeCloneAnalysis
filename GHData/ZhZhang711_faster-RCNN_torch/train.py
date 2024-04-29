import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import Dataset, TestDataset, un_normalize
from Faster_RCNN import Faster_RCNN
from utils.config import config
from utils.visualizer import visdom_bbox
from utils import converter, evals
from train_helper import TrainHelper

# ----- no resource pkg in Windows -----
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def evaluate(testset, faster_rcnn, test_num=10000):
    pred_bbox, pred_label, pred_cls = list(), list(), list()
    gt_bbox, gt_label = list(), list()

    for i, (img, size, gt_bbox_, gt_label_) in tqdm(enumerate(testset)):
        size = [size[0][0].item(), size[1][0].item()]
        pred_bbox_, pred_label_, pred_cls_ = faster_rcnn.predict(img, [size])
        gt_bbox += list(gt_bbox_.numpy())
        gt_label += list(gt_label_)
        pred_bbox += pred_bbox_
        pred_label += pred_label_
        pred_cls += pred_cls_
        if i == test_num:
            break

    result = evals.eval_voc_detection(
        pred_bbox, pred_label, pred_cls,
        gt_bbox, gt_label
    )

    return result


def train(**kwargs):
    config._parse(kwargs)
    dataset = Dataset(config)
    print('loading data')
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=config.num_workers)
    testset = TestDataset(config)
    test_dataloader = DataLoader(testset,
                                 batch_size=1,
                                 num_workers=config.test_num_workers,
                                 shuffle=False,
                                 pin_memory=True)
    faster_rcnn = Faster_RCNN()
    print('constructed Faster-RCNN model')
    train_helper = TrainHelper(faster_rcnn).cuda()
    if config.load_path:
        train_helper.load(config.load_path)
        print('load pretrained model from %s' % config.load_path)
    train_helper.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    # --------------- ---- --- ---- --- ---- lr_ = config.lr
    for epoch in range(config.epoch):
        train_helper.reset_meters()
        for i, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = converter.to_scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            train_helper.train_step(img, bbox, label, scale)

            if (i + 1) % config.plot_every == 0:
                train_helper.vis.multi_plot(train_helper.get_meter_data())
                ori_img_ = un_normalize(converter.to_numpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     converter.to_numpy(bbox_[0]),
                                     converter.to_numpy(label_[0]), )
                train_helper.vis.img('gt_img', gt_img)

                _bbox, _label, _cls = train_helper.faster_rcnn.predict([ori_img_], visualize=True)
                print(_bbox[0].shape)
                from ipdb import set_trace
                set_trace()
                pred_img = visdom_bbox(ori_img_,
                                       converter.to_numpy(_bbox[0]),
                                       converter.to_numpy(_label[0]).reshape(-1),
                                       converter.to_numpy(_cls[0]),)
                train_helper.vis.img('pred_img', pred_img)

                train_helper.vis.text(str(train_helper.rpn_cm.value().tolist()), win='rpn_cm')
                train_helper.vis.img('rcnn_cm', converter.to_tensor(train_helper.rcnn_cm.conf, False).float())
                from ipdb import set_trace
                set_trace()

        eval_result = evaluate(test_dataloader, faster_rcnn, test_num=config.test_num)
        train_helper.vis.plot('test_map', eval_result['map'])
        lr_ = train_helper.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map: {}, loss:{}'.format(str(lr_),
                                                    eval_result['map'],
                                                    str(train_helper.get_meter_data()))
        train_helper.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = train_helper.save(best_map=best_map)
        if epoch == 9:
            train_helper.load(best_path)
            train_helper.faster_rcnn.scale_lr(config.lr_decay)
            lr_ = lr_ * config.lr_decay

        if epoch == 13:
            break


if __name__ == '__main__':
    import fire
    fire.Fire()

