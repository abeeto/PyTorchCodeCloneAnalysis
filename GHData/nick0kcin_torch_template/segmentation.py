import os

import torch.utils.data
from albumentations import *
from torch import device

from history import History
from models import load_model, save_model
from opts import opts
from trainers.segmentation_trainer import SegmentationTrainer  as Trainer
from manager import TrainingManager

if __name__ == '__main__':
    opt = opts().parse()
    logger = TrainingManager(opt.save_dir)
    history = History(opt.save_dir, opt.resume)
    # writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True
    print(opt)
    transforms = {
        "train": Compose([
            ToGray(),
            HorizontalFlip(),
            OneOf([
                Compose([
                    OneOf([
                        ElasticTransform(alpha=200, sigma=35, p=0.5, border_mode=cv2.BORDER_WRAP),
                        OpticalDistortion(border_mode=cv2.BORDER_WRAP)
                    ], p=0.5),
                    RandomSunFlare(flare_roi=(0, 0, 1, 0.25), num_flare_circles_lower=1, num_flare_circles_upper=2,
                                   src_radius=150, p=0.5),

                ]),
                OneOf([
                    GaussianBlur(blur_limit=5),
                    MotionBlur(blur_limit=5),
                ], p=0.5),
            ], p=1),
            CoarseDropout(min_holes=32, max_holes=128, max_width=64, max_height=64, min_width=4, min_height=4, p=0.75),
            GaussNoise(var_limit=(5, 30)),
            OneOf([
                CLAHE(),
                RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma(),
                HueSaturationValue(),
                Equalize(),
            ], p=1),
            Normalize()
        ]
        ),
        "val": Compose([
            PadIfNeeded(608, 608, border_mode=cv2.BORDER_CONSTANT),
            SmallestMaxSize(608),
            Normalize()
        ]
        ),
        "predict": Compose([
            ToGray(),
            LongestMaxSize(608),
            CoarseDropout(min_holes=16, max_holes=64, max_width=32, max_height=32, min_width=4, min_height=4, p=0.75),
            GaussNoise(var_limit=(5, 30)),
            OneOf([
                CLAHE(),
                RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma()
            ], p=1),
            # PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, always_apply=True, p=1),
            Normalize()
        ],
        )
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    np.random.seed(0)

    for it in range(5):
        print({"train": list(set(range(5)) - {it}), "val": [it], "test": [it]})
        losses, loss_weights = logger.loss()
        model = logger.model()
        teacher = logger.teacher()
        params = logger.parameters(model)
        optimizer = logger.optimizer(params)
        lr_schedule = logger.lr_scheduler(optimizer)
        start_epoch = 0
        if opt.load_model != '':
            try:
                model, optimizer, start_epoch, best = load_model(model, opt.load_model,
                                                                 optimizer, opt.resume)
                if not opt.resume and not opt.predict:
                    opt.resume = False
                    opt.load_model = ''
            except:
                try:
                    model, optimizer, start_epoch, best = load_model(model, opt.load_model.replace(".", f"{it}."),
                                                                     optimizer, opt.resume)
                    if not opt.resume and not opt.predict:
                        opt.resume = False
                        opt.load_model = ''
                except:
                    history.reset()
        else:
            history.reset()
        if opt.load_teacher != '' and teacher is not None:
            teacher = load_model(teacher, opt.load_teacher)
        metrics = logger.metric
        trainer = Trainer(model, losses, loss_weights, metrics=metrics, teacher=teacher, optimizer=optimizer,
                          device=opt.device,
                          print_iter=opt.print_iter, num_iter=opt.num_iters, batches_per_update=opt.batches_per_update)
        trainer.set_device(opt.gpus, opt.device)
        loaders = logger.dataloaders(transforms=transforms, folds={"train": list(set(range(5)) - {it}),
                                                                   "val": [it],
                                                                   "predict": [it]})

        if opt.predict:
            trainer.predict_masks(loaders["predict"], "/media/nick/DATA/ame_predicts/", it)
            del loaders
            continue

        if lr_schedule:
            for i in range(start_epoch - 1):
                lr_schedule.step()
            print([group_param["lr"] for group_param in optimizer.param_groups])
        for epoch in range(start_epoch + 1, opt.num_epochs + 1):
            log_dict_val, log_dict_test = None, None
            log_dict_train = trainer.train(epoch, loaders["train"])
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, -1, optimizer)
            if "val" in loaders and opt.val_intervals > 0 and not (epoch % opt.val_intervals):
                with torch.no_grad():
                    log_dict_val = trainer.val(epoch, loaders["val"])

            need_save, timespamp = history.step(epoch, log_dict_train, log_dict_val, log_dict_test)
            if need_save:
                save_model(os.path.join(opt.save_dir, str(timespamp) + '.pth'),
                           epoch, model, log_dict_train["loss"], optimizer)

            if lr_schedule:
                lr_schedule.step()
                print([group_param["lr"] for group_param in optimizer.param_groups])
        os.rename(os.path.join(opt.save_dir, 'model_last.pth'), os.path.join(opt.save_dir, f'model_last{it}.pth'))
        os.rename(os.path.join(opt.save_dir, 'model_best.pth'), os.path.join(opt.save_dir, f'model_best{it}.pth'))
        del loaders
