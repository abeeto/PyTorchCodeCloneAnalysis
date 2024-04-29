import torch


def coarse_network(viewing_dirs,
                   num_coarse_loc,
                   t_i_bin_edges,
                   t_i_gap,
                   ray_origin):
    
    u_c = torch.rand(*list(viewing_dirs[:2]) + 
                     [num_coarse_loc]).to(viewing_dirs)
    
    t_i_coarse = t_i_bin_edges + u_c * t_i_gap
    r_coarse = ray_origin[..., None, :] + t_i_coarse[..., :, None] * viewing_dirs[..., None, :]
    
    return (r_coarse, t_i_coarse)


def fine_network(w_i, 
                 num_fine_locs,
                 t_i_coarse,
                 far_bound,
                 ray_origin,
                 viewing_dirs):
    
    w_i = w_i + 1e-5
    pdfs = w_i / torch.sum(w_i, dim = -1, keepdim = True)
    cdfs = torch.cumsum(pdfs, dim = -1)
    cdfs = torch.cat([torch.zeros_like(cdfs[..., :1]),
                      cdfs[..., :-1]], dim = -1)
    
    uniform_samples = torch.randn(list(cdfs.shape[:-1]) + 
                                  [num_fine_locs]).to(w_i)
    
    #Inverse transform sampling to sample depths
    idxs = torch.searchsorted(cdfs, uniform_samples, right = True)
    t_i_fine_bottom_edges = torch.gather(t_i_coarse, dim = 2, index = idxs - 1)
    idx_clone = idxs.clone()
    max_idx = cdfs.shape[-1]
    idx_clone[idx_clone == max_idx] = max_idx - 1
    
    t_i_fine_top_edges = torch.gather(t_i_coarse, 2, idx_clone)
    t_i_fine_top_edges[idxs == max_idx] = far_bound
    t_i_fine_gap = t_i_fine_top_edges - t_i_fine_bottom_edges
    u_i_fine = torch.rand_like(t_i_fine_gap).to(ray_origin)
    t_i_fine = t_i_fine_bottom_edges + u_i_fine * t_i_fine_gap
    
    (t_i_fine, _) = torch.sort(torch.cat([t_i_coarse, t_i_fine.detach()], dim = -1), dim = -1)
    
    r_fine = ray_origin[..., None, :] + t_i_fine[..., :, None] * viewing_dirs[..., None, :]
    
    return (r_fine, t_i_fine)


def render_volume(model,
                  points,
                  viewing_dirs,
                  chunk_size,
                  t_i):
    
    points_flat = points.reshape((-1, 3))
    viewing_dirs_repeat = viewing_dirs.unsqueeze(2).repeat(1, 1, points.shape[-2], 1)
    viewing_dirs_flat = viewing_dirs_repeat.reshape((-1, 3))
    colors_list, sigma_list = [], []
    
    for chunk in range(0, points_flat.shape[0], chunk_size):
        
        points_batch = points_flat[chunk:chunk+chunk_size]
        viewing_dirs_batch = viewing_dirs_flat[chunk:chunk+chunk_size]
        pred = model(points_batch, viewing_dirs_batch)
        colors_list.append(pred['rgb_color'])
        sigma_list.append(pred['sigma'])
        
    colors = torch.cat(colors_list).reshape(points.shape)
    sigma = torch.cat(sigma_list).reshape(points.shape[:-1])
    
    delta = t_i[..., 1:] - t_i[..., :-1]
    
    one_e10 = torch.Tensor([1e10]).expand(delta[..., 1:].shape)
    delta = torch.cat([delta, one_e10.to(delta)], dim = -1)
    delta = delta * viewing_dirs.norm(dim = -1).unsqueeze(-1)
    
    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(1.0 - alpha + 1e-10, -1)
    
    T = torch.roll(T, 1, -1)
    T[..., 0] = 1.0
    
    w = T * alpha
    pixel_colors = (w[..., None] * colors).sum(dim = -2)
    
    return (pixel_colors, w)


def run_one_iter(viewing_dirs,
                 num_coarse_loc,
                 t_i_bin_edges,
                 t_i_gap,
                 ray_origin,
                 chunk_size,
                 coarse_model,
                 num_fine_locs,
                 far_bound,
                 fine_model):
    
    (r_coarse, t_i_coarse) = coarse_network(viewing_dirs, num_coarse_loc, t_i_bin_edges, t_i_gap, ray_origin)
    (pixel_colors_coarse, w_coarse) = render_volume(coarse_model, r_coarse, viewing_dirs, chunk_size, t_i_coarse)
    
    (r_fine, t_i_fine) = fine_network(w_coarse, num_fine_locs, t_i_coarse, far_bound, ray_origin, viewing_dirs)
    (pixel_colors_fine, _) = render_volume(fine_model, r_fine, viewing_dirs, chunk_size, t_i_fine)
    
    return (pixel_colors_coarse, pixel_colors_fine)

