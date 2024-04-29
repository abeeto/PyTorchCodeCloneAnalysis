from skimage import io
import numpy as np
import torch
import utils as U


def integralImage(x):
    x_s = torch.cumsum(x[:,0,:,:], dim=2)
    y_s = torch.cumsum(x[:,1,:,:], dim=1)
    out = torch.stack( [x_s, y_s], 1)
    return out-1

def repeat(x, n_repeats):
    rep = torch.ones(n_repeats).int().unsqueeze(0)
    x = x.unsqueeze(1).int()
    x = x*rep
    return x.flatten()

def normalize(sampling_grid, height, width):

    minimum_x = sampling_grid[:,0,:,0]
    maximum_x = sampling_grid[:,0,:,-1]

    minimum_y = sampling_grid[:,1,0,:]
    maximum_y = sampling_grid[:,1,-1,:]

    minimum_x = minimum_x.unsqueeze(-1).repeat(1,1,width)
    maximum_x = maximum_x.unsqueeze(-1).repeat(1,1,width)

    minimum_y = minimum_y.unsqueeze(1).repeat(1,height,1)
    maximum_y = maximum_y.unsqueeze(1).repeat(1,height,1)

    norm_x = (sampling_grid[:,0,:,:] - minimum_x) / (maximum_x-minimum_x).float()
    norm_y = (sampling_grid[:,1,:,:] - minimum_y) / (maximum_y-minimum_y).float()

    sampling_grid_norm = torch.stack( ((width-1)*norm_x, (height-1)*norm_y), 1)

    return sampling_grid_norm

def logisticGrowth(x, maxgrad):
    out = maxgrad / (1 + (maxgrad-1)*torch.exp(-x))
    return out


def resample2D(im, sampling_grid, height, width, samples, channels):
    x_s, y_s = sampling_grid[:,0,:,:], sampling_grid[:,1,:,:]

    x = x_s.flatten()
    y = y_s.flatten()

    height_f = float(height)
    width_f = float(width)
    out_height = int(height_f)
    out_width = int(width_f)
    zero = int(0)
    max_y = int(height-1)
    max_x = int(width-1)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    dim2 = width
    dim1 = width*height
    base = repeat(torch.arange(samples)*dim1, out_height*out_width)
    base = U.to_cuda(base)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = torch.reshape(im.permute(0,2,3,1), (-1,3))
    Ia = torch.index_select(im_flat,0,idx_a.long())
    Ib = torch.index_select(im_flat,0,idx_b.long())
    Ic = torch.index_select(im_flat,0,idx_c.long())
    Id = torch.index_select(im_flat,0,idx_d.long())


    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
    wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
    wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
    wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)


    s_out = torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=0)
    output = s_out.sum(dim=0)

    output = output.view(samples, height, width, channels)
    output = output.permute(0,3,1,2)

    return output


def smoothTransformer2D(inp):
    if len(inp) == 3:
        [im, defgrad, affine] = inp # defgrad in range [-1,1] 
    else:
        [im, defgrad] = inp # defgrad in range [-1,1]

    defgrad = logisticGrowth(defgrad, 2.0)

    base_grid = U.to_cuda(integralImage(torch.ones((defgrad.shape[0], defgrad.shape[1], defgrad.shape[2], defgrad.shape[3]))))
    sampling_grid = integralImage(defgrad)

    samples = im.shape[0]
    channels = im.shape[1]
    height = im.shape[2]
    width = im.shape[3]

    try:
        identity = U.to_cuda(torch.cat(samples*[torch.Tensor([[1,0,0,0,1,0,0,0,1]])]))
        affine = affine + identity
        affine = torch.reshape(affine, (samples,3,3))
        sampling_grid = torch.cat( (sampling_grid,U.to_cuda(torch.ones((samples,1,height,width)))), 1)
        sampling_grid = sampling_grid.permute(0,2,3,1)
        sampling_grid = torch.matmul( sampling_grid.view(samples,-1,3), torch.transpose(affine,1,2)) #********
        sampling_grid = sampling_grid.view(samples,height,width,3) #********
        sampling_grid = sampling_grid.permute(0,3,1,2)
        sampling_grid = sampling_grid[:,0:2,:,:]

    except:
        pass

    sampling_grid_norm = normalize(sampling_grid, height, width)

    sampling_grid_inverse = 2*base_grid - sampling_grid_norm
    mov_def = resample2D(im, sampling_grid_norm, height, width, samples, channels)

    ref_def = resample2D(mov_def,sampling_grid_inverse, height, width, samples, channels)

    return mov_def, ref_def, sampling_grid_norm, sampling_grid_inverse
