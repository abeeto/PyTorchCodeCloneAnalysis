import argparse
import time
import os

import torch
import numpy as np
import cv2

from module.classifier import Classifier
from utils.module_select import get_model, get_data_module
from utils.yaml_helper import get_configs


def inference(cfg, ckpt_path):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = get_data_module(cfg['dataset_name'])(
        dataset_dir=cfg['dataset_dir'],
        workers=cfg['workers'],
        batch_size=1,
        input_size=cfg['input_size']
    )
    data_module.prepare_data()
    data_module.setup()

    '''
    Get Model by pytorch lightning
    '''
    model = get_model(cfg['model'])(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])

    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module =Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    '''
    Get Model by pytorch
    '''
    # checkpoint = torch.load(ckpt_path)           
    # state_dict = checkpoint['state_dict']
    # # update keys by dropping `model.`
    # for key in list(state_dict):
    #     state_dict[key.replace('model.', '')] = state_dict.pop(key)
    
    # model = get_model(cfg['model'])(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])
    # model.load_state_dict(state_dict)
    # model.eval()

    # if torch.cuda.is_available:
    #     model = model.to('cuda')


    # Inference
    for sample in data_module.val_dataloader():
        batch_x, batch_y = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
            # predictions = model(batch_x)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        print(f'Label: {batch_y[0]}, Prediction: {torch.argmax(predictions)}')

        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        cv2.imshow('Result', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
