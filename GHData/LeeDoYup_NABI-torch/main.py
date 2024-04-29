import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import utils
import os
import warnings
warnings.filterwarnings("ignore")


from dataset import FeatureDataset 
from train import train, evaluate


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    
    #Training Specifications
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    
    #Model type
    parser.add_argument('--model', type=str, default='baseline')
    
    #Data Directory Path
    parser.add_argument('--data_dir', type=str, default='./data')
    
    # Save model checkpoints in save_dir/save_name.npy
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--save_name', type=str, default='baseline')

    # Save test outputs (probs) in output_dir/output_name.npy
    parser.add_argument('--output_dir', type=str, default='test_preds')    
    parser.add_argument('--output_name', type=str, default='outputs')   

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    batch_size = args.batch_size
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    #Construct dataset
    datasets = {}
    loaders = {}
    for key in ['train', 'eval', 'test']:
        if key == 'train':
            is_shuffle = True
        else:
            is_shuffle = False
        
        datasets[key] = FeatureDataset(key)
        loaders[key] = DataLoader(datasets[key], 
                                  batch_size,
                                  shuffle=is_shuffle, 
                                  num_workers=4)
    
    #Construct Model (by argument)
    constructor = 'build_%s' % args.model
    print("[*] Model Construction Start")
    if args.model == 'baseline':
        import models.baseline as baseline
        print("[*]\t None model construction")
        #model = getattr(baseline.model, constructor)(train_dset, args.num_hid).cuda()
        model = baseline.model()
    else:
        raise NotImplementedError
    
    #For GPU computation
    model = nn.DataParallel(model).cuda()
    save_path = os.path.join(args.save_dir, args.save_name)
    
    '''
    # Train & Evaluate
    '''
    train(model, loaders['train'], loaders['eval'], save_path, args)
    
    '''
    # Test Run
    '''
    test_loss, test_preds, test_roc = evaluate(model, loaders['test'], reload=True, save_path=save_path)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    np.save(os.path.join(args.output_dir, args.output_name)+'.npy', test_preds)