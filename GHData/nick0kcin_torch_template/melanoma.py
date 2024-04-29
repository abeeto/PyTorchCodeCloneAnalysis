import os

import numpy as np
import torch.utils.data
from albumentations import *
from sklearn.metrics import roc_auc_score
from torch import device

from history import History
from models.__init__ import load_model, save_model
from opts import opts
from trainers.classify_trainer import ClassifyTrainer as Trainer
from manager import TrainingManager
from transforms.random_lines import random_microscope, AdvancedHairAugmentation

if __name__ == '__main__':
    opt = opts().parse()
    logger = TrainingManager(opt.save_dir)
    history = History(opt.save_dir, opt.resume)
    torch.backends.cudnn.benchmark = True
    print(opt)
    transforms = {
        "train": Compose([
            IAAAffine(shear=12, p=0.7),
            ShiftScaleRotate(rotate_limit=45, scale_limit=(-0.5, 0.5)),
            Flip(),
            Transpose(),
            ElasticTransform(alpha=100, sigma=25, p=0.5),
            AdvancedHairAugmentation(hairs=10, hairs_folder="hairs"),
            random_microscope(),
            CoarseDropout(min_holes=8, max_width=16, max_height=16, p=0.75),
            OneOf([
                CLAHE(),
                GaussNoise(),
                GaussianBlur(),
                RandomBrightnessContrast(),
                RandomGamma()
            ]),
            Normalize()
        ]
        ),
        "val": Normalize(),
        "test": Normalize(),
        "predict": Compose([
            IAAAffine(shear=12, p=0.7),
            ShiftScaleRotate(rotate_limit=45, scale_limit=(-0.5, 0.5)),
            Flip(),
            Transpose(),
            ElasticTransform(alpha=100, sigma=25, p=0.5),
            AdvancedHairAugmentation(hairs=10, hairs_folder="hairs"),
            random_microscope(),
            CoarseDropout(min_holes=8, max_width=16, max_height=16, p=0.75),
            OneOf([
                CLAHE(),
                GaussNoise(),
                GaussianBlur(),
                RandomBrightnessContrast(),
                RandomGamma()
            ]),
            Normalize()
        ]),
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    np.random.seed(0)
    folds = list(range(0, 15))
    np.random.shuffle(folds)
    print(folds)

    aucs = []
    targets = []
    predicts = []
    for it in range(5):
        print({"train": folds[:3 * it] + folds[3 * it + 3:], "val": folds[3 * it: 3 * it + 3]})
        losses, loss_weights = logger.loss
        model = logger.model
        teacher = logger.teacher
        params = logger.parameters(model)
        optimizer = logger.optimizer(params)
        lr_schedule = logger.lr_scheduler(optimizer)
        start_epoch = 0
        if opt.load_model != '':
            try:
                model, optimizer, start_epoch, best = load_model(model, opt.load_model,
                                                                 optimizer, opt.resume)
            except:
                try:
                    model, optimizer, start_epoch, best = load_model(model, opt.load_model.replace(".", f"{it}."),
                                                                     optimizer, opt.resume)
                except:
                    try:
                        model.load_state_dict(torch.load(opt.load_model.replace(".", f"{it}.")))
                        print("load inf model")
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
        loaders = logger.dataloaders(transforms=transforms, folds={"train": folds[:3 * it] + folds[3 * it + 3:],
                                                                   "val": folds[3 * it: 3 * it + 3],
                                                                   "test": folds[3 * it: 3 * it + 3]})

        if opt.predict:
            try:
                f = open(f"exp/{opt.exp_id}/predict{it}.csv", "r")
            except:
                trainer.predict_file(loaders["predict"], f"exp/{opt.exp_id}/predict{it}.csv")
                if not opt.test:
                    continue
            # continue

        if opt.test:
            pred, gt = trainer.predict_partial(loaders["test"], f"exp/{opt.exp_id}/predict_val{it}.csv")
            targets.extend(gt)
            predicts.extend(pred)
            val = roc_auc_score(targets, predicts)
            print(val)
            continue

        if lr_schedule:
            for i in range(start_epoch):
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

            if "test" in loaders and opt.test_intervals > 0 and not (epoch % opt.test_intervals):
                log_dict_test = trainer.test(loaders["val"])

            need_save, timespamp = history.step(epoch, log_dict_train, log_dict_val, log_dict_test)
            if need_save:
                save_model(os.path.join(opt.save_dir, str(timespamp) + '.pth'),
                           epoch, model, log_dict_train["loss"], optimizer)

            if lr_schedule:
                lr_schedule.step(log_dict_test)
                print([group_param["lr"] for group_param in optimizer.param_groups])
        os.rename(os.path.join(opt.save_dir, 'model_last.pth'), os.path.join(opt.save_dir, f'model_last{it}.pth'))
        os.rename(os.path.join(opt.save_dir, 'model_best.pth'), os.path.join(opt.save_dir, f'model_best{it}.pth'))
        aucs.append(history.best[0])

    val = roc_auc_score(targets, predicts)
    print(val)
