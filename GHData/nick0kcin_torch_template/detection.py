import os
import torch.utils.data
from torch import device
from opts import opts
from models.__init__ import  load_model, save_model
from trainers.obj_det_kp_trainer import ObjDetKPTrainer as Trainer
from history import History
from manager import TrainingManager
from torch.utils.tensorboard import SummaryWriter
from albumentations import *
import cv2
import json
from pycocotools.cocoeval import COCOeval

try:
    from apex import amp
    APEX = True
except ModuleNotFoundError:
    APEX = False


if __name__ == '__main__':
    opt = opts().parse()
    logger = TrainingManager(opt.save_dir)
    history = History(opt.save_dir, opt.resume)
    writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True
    print(opt)
    transforms = {
        "train":  Compose([
            ShiftScaleRotate(rotate_limit=90, scale_limit=(-0.35, 0.3),
                             border_mode=cv2.BORDER_CONSTANT),
            PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            RandomCrop(512, 512, always_apply=True),
            Flip(),
            Transpose(),
            # ElasticTransform(alpha=250, sigma=30, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            ImageCompression(quality_lower=80, always_apply=True),
            # CoarseDropout(max_holes=40, min_holes=6),
            ToGray(),
            OneOf([
                CLAHE(),
                MotionBlur(),
                RGBShift(),
                RandomBrightnessContrast()
                ]),
            Normalize()
            ],  bbox_params=BboxParams(format='pascal_voc', min_area=80,
                                               min_visibility=0.5, label_fields=['super_label'])),
        "val": Compose([
            # PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_CONSTANT),
            Normalize()
        ],  bbox_params=BboxParams(format='pascal_voc', min_area=100,
                                               min_visibility=0.5, label_fields=['super_label'])),
        "test": Compose([
            # PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_CONSTANT),
            ToGray(),
            Normalize()
        ],  bbox_params=BboxParams(format='pascal_voc', min_area=100,
                                               min_visibility=0.0, label_fields=['super_label'])),
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    loaders = logger.dataloaders(transforms=transforms)
    losses, loss_weights = logger.loss()
    model = logger.model()
    teacher = logger.teacher()
    params = logger.parameters(model)
    optimizer = logger.optimizer(params)
    lr_schedule = logger.lr_scheduler(optimizer)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch, best = load_model(model, opt.load_model, optimizer, opt.resume)
    if opt.load_teacher != '' and teacher is not None:
        teacher = load_model(teacher, opt.load_teacher)
    metrics = logger.metric
    trainer = Trainer(model, losses, loss_weights,  metrics=metrics, teacher=teacher, optimizer=optimizer, device=opt.device,
                      print_iter=opt.print_iter, num_iter=opt.num_iters, batches_per_update=opt.batches_per_update,
                      **logger.trainer_params())
    trainer.set_device(opt.gpus, opt.device)
    coco = loaders["test"].dataset.coco()

  
    if lr_schedule:
        for i in range(start_epoch):
            lr_schedule.step()
        print([group_param["lr"] for group_param in optimizer.param_groups])
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        log_dict_val, log_dict_test = None, None
        log_dict_train = trainer.train(epoch, loaders["train"])
        writer.add_scalars("train", log_dict_train, 1)
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, -1, optimizer)
        if "val" in loaders and opt.val_intervals > 0 and not(epoch % opt.val_intervals):
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, loaders["val"])
            writer.add_scalars("val", log_dict_val, opt.val_intervals)

        if "test" in loaders and opt.test_intervals > 0 and not(epoch % opt.test_intervals):
            log_dict_test = trainer.test(loaders["test"])
            writer.add_scalars("test", log_dict_test, opt.test_intervals)

        need_save, timespamp = history.step(epoch, log_dict_train, log_dict_val, log_dict_test)
        if need_save:
            save_model(os.path.join(opt.save_dir, str(timespamp) + '.pth'),
                       epoch, model, log_dict_train["loss"], optimizer)

        if lr_schedule:
            lr_schedule.step()
            print([group_param["lr"] for group_param in optimizer.param_groups])
    writer.close()