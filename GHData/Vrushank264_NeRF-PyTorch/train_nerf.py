import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from nerf import NeRF
from functions import *
import configs


def main():
    
    torch.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    writer = SummaryWriter(configs.log_dir)
    
    '''Load Data'''
    data = np.load(configs.path)
    imgs = data['images'] / 255.0
    poses = data['poses']
    focal = float(data['focal'])
    cam_distance = float(data['camera_distance'])
    
    
    '''Get initial viewing directions and ray origin
       They are same across all the samples, but we 
       just rotate them according to the orientation 
       of the camera.'''

    img_size = imgs.shape[1]
    xs = torch.arange(img_size) - img_size / 2
    ys = torch.arange(img_size) - img_size / 2
    (xs, ys) = torch.meshgrid(xs, -ys, indexing = 'xy')
    pixel_coords = torch.stack([xs, ys, -focal * torch.ones_like(xs)], dim = -1)
    
    camera_coords = pixel_coords / focal
    initial_viewing_dir = camera_coords.to(configs.device)
    initial_ray_origin = torch.Tensor(np.array([0, 0, cam_distance])).to(configs.device)
    
    '''Monitor Training sample'''
    monitor_idx = 111
    monitor_img = torch.Tensor(imgs[monitor_idx]).to(configs.device)
    monitor_rot = torch.Tensor(poses[monitor_idx, :3, :3]).to(configs.device)
    monitor_viewing_dirs = torch.einsum("ij, hwj -> hwi", monitor_rot, initial_viewing_dir)
    monitor_ray_origin = (monitor_rot @ initial_ray_origin).expand(monitor_viewing_dirs.shape)
    
    '''Test render view'''
    test_idx = 380
    test_img = torch.Tensor(imgs[test_idx]).to(configs.device)
    test_rot = torch.Tensor(poses[test_idx, :3, :3]).to(configs.device)
    test_viewing_dirs = torch.einsum("ij, hwj -> hwi", test_rot, initial_viewing_dir)
    test_ray_origin = (test_rot @ initial_ray_origin).expand(test_viewing_dirs.shape)
    
    
    '''Stratified Sampling for sampling depths'''
    t_i_gap = (configs.far_bound - configs.near_bound) / configs.num_coarse_samples
    t_i_bin_edges = (configs.near_bound + torch.arange(configs.num_coarse_samples) * t_i_gap).to(configs.device)
    
    
    '''Training'''
    coarse_model = NeRF().to(configs.device)
    fine_model = NeRF().to(configs.device)
    opt = torch.optim.Adam(list(coarse_model.parameters()) + list(fine_model.parameters()), 
                           lr = configs.LR)
    
    criterion = nn.MSELoss()
    
    train_idxs = np.arange(len(imgs)) != test_idx
    imgs = torch.Tensor(imgs[train_idxs])
    poses = torch.Tensor(poses[train_idxs])
    num_pixels = img_size ** 2
    pixels = torch.full((num_pixels, ), 1/ num_pixels).to(configs.device)
    writer.add_image('Validation/target', monitor_img, 0)
    writer.add_image('Test/target', test_img, 0)
    
    coarse_model.train()
    fine_model.train()
    
    #loop = tqdm(train_idxs, leave = True, position = 0)
    
    for idx in range(configs.num_iters):
        
        target_img_idx = np.random.randint(imgs.shape[0])
        target_pose = poses[target_img_idx].to(configs.device)
        rot = target_pose[:3, :3]
        
        viewing_dirs = torch.einsum("ij, hwj->hwi", rot, initial_viewing_dir)
        ray_origin = (rot @ initial_ray_origin).expand(viewing_dirs.shape)
        
        pixel_idxs = pixels.multinomial(configs.num_pixel_batch, False)
        pixel_idxs_rows = pixel_idxs // img_size
        pixel_idxs_cols = pixel_idxs % img_size
        viewing_dirs_batch = viewing_dirs[pixel_idxs_rows, pixel_idxs_cols].reshape(
                                                                   configs.batch_size,
                                                                   configs.batch_size, 
                                                                   -1)
        ray_origin_batch = ray_origin[pixel_idxs_rows, pixel_idxs_cols].reshape(configs.batch_size,
                                                                                configs.batch_size, 
                                                                                -1)
        
        (pixel_colors_coarse, pixel_colors_fine) = run_one_iter(viewing_dirs_batch, 
                                                                configs.num_coarse_loc, 
                                                                t_i_bin_edges, 
                                                                t_i_gap, 
                                                                ray_origin_batch, 
                                                                configs.chunk_size, 
                                                                coarse_model, 
                                                                configs.num_fine_locs, 
                                                                configs.far_bound, 
                                                                fine_model)
        
        target_img = imgs[target_img_idx].to(configs.device)
        target_img_batch = target_img[pixel_idxs_rows, pixel_idxs_cols].reshape(pixel_colors_fine.shape)
        
        coarse_loss = criterion(pixel_colors_coarse, target_img_batch)
        fine_loss = criterion(pixel_colors_fine, target_img_batch)
        total_loss = coarse_loss + fine_loss
        psnr = -10.0*torch.log10(total_loss)
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        if idx % configs.display_on_tensorboard == 0:
            
            print(idx)
            coarse_model.eval()
            fine_model.eval()
            
            with torch.no_grad():
                
                (pix_colors_coarse_monitor, pix_colors_fine_monitor) = run_one_iter(monitor_viewing_dirs,
                                                                                    configs.num_coarse_loc,
                                                                                    t_i_bin_edges,
                                                                                    t_i_gap,
                                                                                    monitor_ray_origin,
                                                                                    configs.chunk_size,
                                                                                    coarse_model,
                                                                                    configs.num_fine_locs,
                                                                                    configs.far_bound,
                                                                                    fine_model)
                monitor_loss = criterion(pix_colors_fine_monitor, monitor_img)
                monitor_psnr = -10.0 * torch.log10(monitor_loss)
                
                (pix_colors_coarse_test, pix_colors_fine_test) = run_one_iter(test_viewing_dirs,
                                                                             configs.num_coarse_loc,
                                                                             t_i_bin_edges,
                                                                             t_i_gap,
                                                                             test_ray_origin,
                                                                             configs.chunk_size,
                                                                             coarse_model,
                                                                             configs.num_fine_locs,
                                                                             configs.far_bound,
                                                                             fine_model)
                
                test_loss = criterion(pix_colors_fine_test, test_img)
                test_psnr = -10.0 * torch.log10(test_loss)
            
            writer.add_scalar('Validation/loss', monitor_loss.item(), idx)
            writer.add_scalar('Validation/PSNR', monitor_psnr.item(), idx)
            writer.add_scalar('Test/loss', test_loss.item(), idx)
            writer.add_scalar('Test/PSNR', test_psnr.item(), idx)
            
            writer.add_image('Validation/pred', pix_colors_fine_monitor, idx)
            writer.add_image('Test/pred', pix_colors_fine_test, idx)
            torch.save(fine_model.state_dict(), f'{configs.save_path}/NeRF_model.pth')
 
            
        writer.add_scalar('Training/Total_loss', total_loss.item(), idx)
        writer.add_scalar('Training/PSNR', psnr.item(), idx)
            
    
    
    print("Done")


if __name__ == '__main__':

    main()    

