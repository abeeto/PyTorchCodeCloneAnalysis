import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')

from dataset import yoloPascal
from torch.utils.data import DataLoader
from models.yolo_head import Yolodet
from config import hrnet_yolo, dark53_yolo
from utils.checkpoint import load_checkpoint
from utils.util import nms
import os
import cv2
import numpy as np
import time
cfg=dark53_yolo

loader = DataLoader(yoloPascal(cfg, 'val'), batch_size=1, shuffle=False)
model = Yolodet(cfg, pretrained=False)
load_checkpoint(model, 'weights/dark_yolo/model_13.pth')
model.eval()
model.cuda()
for batch in loader:
    start= time.time()
    out = model(batch['input'].to(device='cuda'))
    print('fps: {}'.format(1/(time.time()-start)))
    file_name =  loader.dataset.coco.loadImgs(ids=[batch['img_id'].numpy()[0]])[0]['file_name']
    img_path = os.path.join( loader.dataset.img_dir, file_name)
    image = cv2.imread(img_path)
    shape = image.shape
    bboxes = model.convert_pred(out, shape , 0.5)
    bboxes = nms(bboxes,0.5,0.5)
    for bb in bboxes:
        x, y, x1, y1 = bb.astype(np.int)[:4]
        cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 3)
    cv2.imshow('', image)
    if cv2.waitKey(0) & 0xff == 27:
        break
cv2.destroyAllWindows()