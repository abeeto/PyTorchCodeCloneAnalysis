import argparse
import os

import torch
import torch.nn as nn

import models as models
import models_cpu as models_cpu
from rank_selection.select_ranks import estimate_ranks_upper_bound

parser = argparse.ArgumentParser(description="submit for layer decomposition")
parser.add_argument('--eps', default='0.002', type=float, required=False, help='(default=%(default)s)')
parser.add_argument('--device', default='cuda', type=str, required=False, help='device to use for decompositions')
parser.add_argument('--dpath', default='', type=str, required=True, help='Dataset Directory')
parser.add_argument('--mpath', default='', type=str, required=True, help='Trained Model file(.pth)')
parser.add_argument('--tlabels', default='', type=str, required=True, help='Labels for Train Set')
parser.add_argument('--vlabels', default='', type=str, required=True, help='Labels for Test/validation Set')
parser.add_argument('--ranks_dir', default='../ranks_PA-100K/', type=str, required=False, help='Dir to save ranks and Tensors,will be auto-created')
parser.add_argument('--curr_dir', default='/trinity/home/a.jha/a.jha/scripts/', type=str, required=False, help='Dir to current scripts')
parser.add_argument('--factors', default='both', type=str, required=True, help='SVD or CPD-EPC or both')
parser.add_argument('--attr_num', default='26', type=int, required=True, help='(35)PETA or (26)PA-100K')
parser.add_argument('--experiment', default='PA-100K', type=str, required=True, help='Type of experiment PETA or PA-100K')
args = parser.parse_args()


path = args.curr_dir
os.chdir(path)
device = args.device
decomposition = args.factors
attr_num = args.attr_num
nx = 1

if device == "cpu":
    model = models_cpu.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
else:
    model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)
    model = torch.nn.DataParallel(model).cuda()


if decomposition == "SVD" or decomposition == "svd":
    lnames_to_compress = [module_name 
                          for module_name, module in model.named_modules() 
                          if isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1]
    # get maximum ranks for decomposition
    max_ranks = estimate_ranks_upper_bound(model, lnames_to_compress, nx=nx,
                                           input_img_size=(1, 3, 256, 128), device=device)
    for items in max_ranks:
        if device == "cpu":
#              command = """sbatch run_cpu.sh layer_decomposition.py --layer={0} --rank={1} --eps={eps} --device={device} --dpath={dpath}""".format(items, max_ranks[items], **vars(args))
            command = """sbatch run_cpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        else:
            command = """sbatch run_gpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        os.system(command)
#         print(command)
#         break

elif decomposition == "CPD-EPC" or decomposition == "cpd-epc":
    # create list of layer names to be compressed
    lnames_to_compress = [module_name 
                          for module_name, module in model.named_modules() 
                          if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1]
    # get maximum ranks for decomposition
    max_ranks = estimate_ranks_upper_bound(model, lnames_to_compress, nx=nx,
                                           input_img_size=(1, 3, 256, 128), device=device)
    
    for items in max_ranks:
        if device == "cpu":
            command = """sbatch run_cpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        else:
            command = """sbatch run_gpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        os.system(command)
#         break
#         print(command)

        
else:
    # create list of layer names to be compressed
    lnames_to_compress = [module_name 
                          for module_name, module in model.named_modules() 
                          if isinstance(module, nn.Conv2d) and module.kernel_size[0] >= 1]
    # get maximum ranks for decomposition
    max_ranks = estimate_ranks_upper_bound(model, lnames_to_compress, nx=nx,
                                           input_img_size=(1, 3, 256, 128), device=device)
    for items in max_ranks:
        if device == "cpu":
            command = """sbatch run_cpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        else:
            command = """sbatch run_gpu.sh layer_decomposer.py --layer={0} --rank={1} --eps={2} --device={3} --dpath={4} --mpath={5} --tlabels={6} --vlabels={7} --ranks_dir={8} --experiment={9} --attr_num={10}""".format(items, max_ranks[items], args.eps, args.device, args.dpath, args.mpath, args.tlabels, args.vlabels, args.ranks_dir, args.experiment, args.attr_num)
        os.system(command)
#         break
#         print(command)
    
