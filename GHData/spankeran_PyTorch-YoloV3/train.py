#-*-coding:utf-8-*-
import sys
import os.path as osp
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader

BASE_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(BASE_DIR, 'data'))
sys.path.append(osp.join(BASE_DIR, 'models'))
sys.path.append(osp.join(BASE_DIR, 'utils'))

from models.yolo_network import yolo_network
from data.dataset import VOCDataset
from test import evaluate
import utils.logger
import utils.augmentation
import utils.utils

logger = utils.logger.logger()

dataset_list = {
    'VOC': VOCDataset,
    'COCO': None
}

def train() :
    parser = argparse.ArgumentParser(
        prog='Pytorch-YoloV3',
        description='a simple implementation of YoloV3 in PyTorch'
    )
    parser.add_argument("--epochs", "-e", type=int, default=120, help="number of epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="the size each batch")
    parser.add_argument("--subdivisions", "-s", type=int, default=8, help="the size of subdivided batch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="original learning rate")
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4, help="weight decay in optimizer")
    parser.add_argument("--burnin", "-b", type=int, default=1000, help="iters to warm up")
    parser.add_argument("--decay_step", "-ds", type=tuple, default=(100,110), help="learning rate decay steps")
    parser.add_argument("--model_name", "-mn", type=str, default="yolov3-spp", help="the name of model")
    parser.add_argument("--pretrain_weights_name", "-pwn", type=str, default="darknet53.conv.74",
                        help="the name of weights file name")
    parser.add_argument("--load_pretrain_weights", "-lpw", type=utils.utils.str2bool, default="t",
                        help="if True load pretrain weights")
    parser.add_argument("--continue_train", "-ct", type=utils.utils.str2bool, default="t",
                        help="if True continue last train")
    parser.add_argument("--ignore_thresold", type=float, default=0.5, help="ignore thresold")
    parser.add_argument("--nms_thresold", type=float, default=0.4, help="nms thresold")
    parser.add_argument("--conf_thresold", type=float, default=0.5, help="conf thresold")
    parser.add_argument("--coord_scale", "-cs", type=float, default=5.0, help="the weight of coordinate loss")
    parser.add_argument("--conf_scale", "-fs", type=float, default=1.0, help="the weight of confidence loss")
    parser.add_argument("--cls_scale", "-cls", type=float, default=1.0, help="the weight of classification loss")
    parser.add_argument("--img_size", type=int, default=416, help="input size of images")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument('-dt', '--dataset_type', type=str, default='VOC', choices=('VOC', 'COCO'), help='dataset type')
    parser.add_argument("--train_dataset", type=tuple, default=(('2007', 'trainval'), ('2012', 'trainval')),
                        help="train dataset")
    parser.add_argument("--evaluate_dataset", type=tuple, default=(('2007', 'test'),), help="evaluate dataset")
    parser.add_argument("--weights_dirname", "-wp", type=str, default="weights", help="dirname of weights files")
    parser.add_argument("--results_dirname", "-rp", type=str, default="results", help="dirname of results files")
    parser.add_argument("--checkpoints_dirname", "-cp", type=str, default="checkpoints",
                        help="dirname of checkpoints files")
    opt = parser.parse_args()

    dataset = dataset_list[opt.dataset_type](
        setname=opt.train_dataset,
        transform=utils.augmentation.data_augmentation
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=opt.batch_size // opt.subdivisions, shuffle=True, num_workers=2,
        collate_fn=dataset.collate_fn(), pin_memory=True
    )

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = yolo_network(
        network_name=osp.join(BASE_DIR, 'models', 'cfg', '%s.cfg' % opt.model_name), class_num=dataset.get_class_num(),
        ignore_thresold=opt.ignore_thresold, conf_thresold=opt.conf_thresold, nms_thresold=opt.nms_thresold,
        coord_scale=opt.coord_scale, conf_scale=opt.conf_scale, cls_scale=opt.cls_scale,
        device=device
    ).to(device)
    model.apply(utils.utils.weight_init)
    if opt.load_pretrain_weights :
        weights_path = osp.join(BASE_DIR, utils.utils.checkdir(opt.weights_dirname), opt.pretrain_weights_name)
        model.load_weights(weights_path)

    optimizer = optim.SGD(
        params=model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.decay_step, gamma=0.1)
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=140, eta_min=1e-6)

    start_epoch = 0
    train_infos = utils.logger.train_infos()
    result_list = utils.logger.train_results()

    if opt.continue_train:
        ckpt_name = utils.utils.get_latest_file(osp.join(BASE_DIR, utils.utils.checkdir(opt.checkpoints_dirname)))
        if ckpt_name :
            ckpt_path = osp.join(BASE_DIR, opt.checkpoints_dirname, ckpt_name)
            load_infos = utils.utils.load_train_ckpt([model, optimizer], ckpt_path, device)
            start_epoch = load_infos[0] + 1
            result_list.__infos__ = load_infos[1]
            logger.info("resume from epoch: %d" % start_epoch)
        else :
            logger.error("no checkpoints files exist, start training from epoch: 1")

    else:
        logger.info("start training from epoch: 1")

    for epoch in range(start_epoch + 1, opt.epochs + 1) :
        model.train()
        train_infos.reset()

        for iter, (imgs, targets, _) in enumerate(dataloader) :

            imgs = imgs.to(device)
            targets = targets.to(device)

            loss = model(imgs, targets, train_infos)
            loss.backward()

            if (iter + 1) % opt.subdivisions == 0 :
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Epoch: [{:03d}/{:03d}]  Iter: [{:05d}/{:05d}] Learning rate: {:.5f}'.format(
                    epoch, opt.epochs, iter + 1, len(dataloader), lr_scheduler.get_lr()[0]) + '\n' + train_infos.get_str())

            if epoch == 1 and iter < opt.burnin :
                utils.utils.warm_up_lr(opt.learning_rate, iter + 1, optimizer, opt.burnin)

        lr_scheduler.step()
        for k, v in train_infos.__infos__.items() :
            result_list.update(k, v.avg)

        if epoch % opt.checkpoint_interval == 0:
            ckpt_path = osp.join(BASE_DIR, opt.checkpoints_dirname, 'checkpoints_epoch_%d.ckpt' % epoch)
            utils.utils.save_train_ckpt([model, optimizer], epoch, result_list.__infos__, ckpt_path)

        if epoch % opt.evaluation_interval == 0 :
            logger.info("evaluate")
            eval_info = evaluate(model, opt.dataset_type, opt.evaluate_dataset, 100, opt.batch_size // opt.subdivisions, device)
            logger.info(eval_info)

    logger.info("Saving train weight...")
    torch.save(model.state_dict(),
        osp.join(BASE_DIR, opt.weights_dirname, "{}_state_{}.pth".format(opt.model_name, utils.utils.time2str()))
    )

    # draw infos curve
    utils.utils.checkdir(osp.join(opt.results_dirname, 'curve'))
    result_list.draw(osp.join(opt.results_dirname, 'curve', 'train_info_{}.png'.format(utils.utils.time2str())))

if __name__ == '__main__' :
    try :
        train()
    except :
        logger.error("error during training!")