import os

import cv2
import numpy as np
import torch.utils.data
from albumentations import Normalize
from torch import Tensor
from torch import device

from history import History
from models import load_model
from opts import opts
from trainers.obj_det_kp_trainer import ObjDetKPTrainer as Trainer
from manager import TrainingManager

if __name__ == '__main__':
    opt = opts().parse()
    video = cv2.VideoCapture(opt.video)
    sc = 0.6
    writer = cv2.VideoWriter(opt.video.replace(".avi", "result.avi"), cv2.VideoWriter_fourcc(*'MJPG'), 25,
                             (int(int(video.get(3) * sc) // 32 * 32), int(int(video.get(4) * sc) // 32 * 32))
                             )
    writer_map = cv2.VideoWriter(opt.video.replace(".avi", "result_map.avi"),
                                 cv2.VideoWriter_fourcc("M", "J", "P", "G"), 25,
                                 (int(int(video.get(3) * sc) // 32 * 32), int(int(video.get(4) * sc) // 32 * 32 * 2))
                                 )
    logger = TrainingManager(opt.save_dir)
    history = History(opt.save_dir, opt.resume)
    torch.backends.cudnn.benchmark = True
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    losses, loss_weights = logger.loss
    model = logger.model
    params = logger.parameters(model)
    optimizer = logger.optimizer(params)
    lr_schedule = logger.lr_scheduler(optimizer)
    model, optimizer, start_epoch, best = load_model(model, opt.load_model, optimizer, opt.resume)
    metrics = logger.metric
    trainer = Trainer(model, losses, loss_weights, metrics=metrics, optimizer=optimizer, device=opt.device,
                      print_iter=opt.print_iter, num_iter=opt.num_iters, batches_per_update=opt.batches_per_update,
                      **logger.trainer_params)
    trainer.set_device(opt.gpus, opt.device)
    for i in range(25 * 1000):
        ret, img = video.read()
        if not ret:
            break
        image = cv2.resize(img, (0, 0), fx=sc, fy=sc)
        image = image[:(image.shape[0] // 32) * 32,
                :(image.shape[1] // 32) * 32, :].copy()
        mini_dataset = torch.utils.data.TensorDataset(
            Tensor(Normalize()(image=image)["image"].transpose(2, 0, 1)).unsqueeze(0))
        mini_dataset.scales = (1,)
        mini_loader = torch.utils.data.DataLoader(mini_dataset)
        img, mp = trainer.visualize(mini_loader)
        img = img | image
        writer.write(img)
        m_img = np.tile((mp * 255).astype(np.uint8), (3, 1, 1)).transpose(1, 2, 0)
    writer.release()
