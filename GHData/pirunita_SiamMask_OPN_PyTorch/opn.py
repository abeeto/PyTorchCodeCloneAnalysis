
import argparse
import cv2
import glob
import os
#import sys
import torch

import numpy as np
from PIL import Image

from OnionPeel.helpers import *
from OnionPeel.OPN import OPN
from OnionPeel.TCN import TCN

parser = argparse.ArgumentParser(description='PyTorch Onion-Peel Demo')

parser.add_argument('--name', type=str, default='Onion-Peel')
parser.add_argument('--opn_path', type=str, default='./OnionPeel/OPN.pth')
parser.add_argument('--tcn_path', type=str, default='./OnionPeel/TCN.pth')

parser.add_argument('--base_path', type=str, default='./data/tennis', help='datasets')
parser.add_argument('--root_path', default='./results/')
parser.add_argument('--result_path', default='./results/tennis/')
parser.add_argument('--mask_path', default='./results/tennis/masks/')
parser.add_argument('--mask_path2', default='./results/tennis/masks2/')
parser.add_argument('--save_path', default='./results/tennis/final/')
parser.add_argument('--scale_factor', type=float, default=1)
parser.add_argument('--dilate_factor', type=int, default=3)

args = parser.parse_args()

if __name__ == '__main__':
    # Make directory
    if not os.path.exists(args.mask_path2):
        os.makedirs(args.mask_path2)
        
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Make directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    img_list = glob.glob(os.path.join(args.base_path, '*.jp*'))
    img = Image.open(img_list[0]).convert('RGB')
    ori_W, ori_H = img.size
    T = len(img_list)
    W = int(args.scale_factor * ori_W)
    H = int(args.scale_factor * ori_H)
        
    frames = np.empty((T, H, W, 3), dtype=np.float32)
    holes = np.empty((T, H, W, 1), dtype=np.float32)
    dists = np.empty((T, H, W, 1), dtype=np.float32)
    for i in range(T-1):
        # rgb
        img_file = os.path.join(args.base_path, '{:05d}.jpg'.format(i+1))
        raw_frame = np.array(Image.open(img_file).convert('RGB').resize((W, H), Image.BILINEAR))/255.
        
        frames[i] = raw_frame
        # mask
        mask_file = os.path.join(args.mask_path, '{:05d}.png'.format(i+1))
        raw_mask = np.array(Image.open(mask_file).resize((W, H), Image.NEAREST).convert('P'), dtype=np.uint8)
        raw_mask = (raw_mask > 0.5).astype(np.uint8)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(args.dilate_factor, args.dilate_factor)))
        holes[i,:,:,0] = raw_mask.astype(np.float32)
        
        # dist
        dists[i,:,:,0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
        
    frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
    holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
    dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
    
    frames = frames * (1-holes) + holes*torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    valids = 1-holes
    
    frames = frames.unsqueeze(0)
    holes = holes.unsqueeze(0)
    dists = dists.unsqueeze(0)
    valids = valids.unsqueeze(0)
    
    # Load model
    opn = nn.DataParallel(OPN())
    tcn = nn.DataParallel(TCN())
    if torch.cuda.is_available():
        opn.cuda()
        tcn.cuda()
    
    opn.load_state_dict(torch.load(os.path.join(args.opn_path)), strict=False)
    tcn.load_state_dict(torch.load(os.path.join(args.tcn_path)), strict=False)
    opn.eval()
    tcn.eval()
    
    # inference
    MEM_EVERY = 5 # every 5 frame as memory frames
    comps = torch.zeros_like(frames)
    ppeds = torch.zeros_like(frames)
    
    # memory encoding 
    midx = list( range(0, T, MEM_EVERY) )
    with torch.no_grad():
        mkey, mval, mhol = opn(frames[:,:,midx], valids[:,:,midx], dists[:,:,midx])

    for f in range(T):
        # memory selection
        if f in midx:
            ridx = [i for i in range(len(midx)) if i != int(f/MEM_EVERY)]
        else:
            ridx = list(range(len(midx)))

        fkey, fval, fhol = mkey[:,:,ridx], mval[:,:,ridx], mhol[:,:,ridx]
        # inpainting..
        for r in range(999): 
            if r == 0:
                comp = frames[:,:,f]
                dist = dists[:,:,f]
            with torch.no_grad(): 
                comp, dist = opn(fkey, fval, fhol, comp, valids[:,:,f], dist)
            
            # update
            comp, dist = comp.detach(), dist.detach()
            if torch.sum(dist).item() == 0:
                break
            
        comps[:,:,f] = comp

    # post-processing...
    ppeds[:,:,0] = comps[:,:,0]
    hidden = None
    for f in range(T):
        with torch.no_grad():
            pped, hidden =\
                    tcn(ppeds[:,:,f-1], holes[:,:,f-1], comps[:,:,f], holes[:,:,f], hidden)
            ppeds[:,:,f] = pped
            
    for f in range(T):
        est = (ppeds[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)
        true = (frames[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8) # h,w,3
        mask = (dists[0,0,f].detach().cpu().numpy() > 0).astype(np.uint8) # h,w,1
        ov_true = overlay_davis(true, mask, colors=[[0,0,0],[0,100,100]], cscale=2, alpha=0.4)

        canvas = np.concatenate([est], axis=0)
        canvas = Image.fromarray(canvas)
        canvas.save(os.path.join(args.save_path, '{:05d}.jpg'.format(f)))
        
        canvas2 = np.concatenate([ov_true], axis=0)
        canvas2 = Image.fromarray(canvas2)
        canvas2.save(os.path.join(args.mask_path2, '{:05d}.jpg'.format(f)))
    
    print('Results are saved: {}'.format(args.save_path))
    print('Masks2 are saved : {}'.format(args.mask_path2))
    
    