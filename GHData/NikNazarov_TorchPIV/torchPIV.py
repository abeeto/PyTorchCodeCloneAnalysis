import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Generator, Tuple
from torch.utils.data import Dataset
from scipy import interpolate
import cv2
import fastSubpixel
from time import time
from PlotterFunctions import natural_keys

class DeviceMap:
    devicies = {
        torch.cuda.get_device_name(i) : torch.device(i)  
        for i in range(torch.cuda.device_count())
    }
    devicies["cpu"] = torch.device("cpu")


def free_cuda_memory():
    # torch.cuda.synchronize()
    if torch.cuda.is_available(): torch.cuda.empty_cache() 

def load_pair(name_a: str, name_b: str, transforms) -> Tuple[torch.Tensor]:
    """
    Helper method, Can be used later in right on fly version of PIV
    Reads image pair from disk as numpy array and performs transforms on it
    """
    try:
        frame_b = cv2.imread(name_b, cv2.IMREAD_GRAYSCALE)
        frame_a = cv2.imread(name_a, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Invalid File Path!")
        return None
    if transforms:
        frame_a = transforms(frame_a)
        frame_b = transforms(frame_b)
    return frame_a, frame_b

class ToTensor:
    """
    Basic transform class. Converts numpy array to torch.Tensor with dtype
    """
    def __init__(self, dtype:  type) -> None:
        self.dtype = dtype
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=self.dtype)

class PIVDataset(Dataset):
    def __init__(self, folder, file_fmt, transform=None):
        self.transform = transform
        filenames = [os.path.join(folder, name) for name 
            in os.listdir(folder) if name.endswith(file_fmt)]
        filenames.sort(key=natural_keys)
        self.img_pairs = list(zip(filenames[::2], filenames[1::2]))
        # self.img_pairs = list(zip(filenames[:-1], filenames[1:]))
    def __len__(self):
        return len(self.img_pairs)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:        

        if torch.is_tensor(index):
            index = index.tolist()

        pair = self.img_pairs[index]
        #imread function works only with latin file path
        img_b = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        img_a = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b

def moving_window_array(array: torch.Tensor, window_size, overlap) -> torch.Tensor:
    """
    This is a nice numpy and torch trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.

    """
    shape = array.shape
    strides = (
        shape[-1] * (window_size - overlap),
        (window_size - overlap),
        shape[-1],
        1
    )
    shape = (
        int((shape[-2] - window_size) / (window_size - overlap)) + 1,
        int((shape[-1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )
    return torch.as_strided(
        array, size=shape, stride=strides 
    ).reshape(-1, window_size, window_size)

def correalte_fft(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cross correlation based on fft method
    Between two torch.Tensors of shape [c, width, height]
    fft performed over last two dimensions of tensors
    """
    corr = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(images_a).conj() *
                               torch.fft.rfft2(images_b)), dim=(-2, -1))
    return corr


def find_first_peak_position(corr: torch.Tensor) -> torch.Tensor:
    """Return Tensor (c, 2) of peak coordinates"""
    c, d, k = corr.shape
    m = corr.view(c, -1).argmax(-1, keepdim=True)
    return torch.cat((m // d, m % k), -1)

def interpolate_nan(
        vec: np.ndarray,
        method: str = 'linear',
        fill_value: int = 0
    ) -> np.ndarray:
    """
    :param vec (:, :): 2D field
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is linear.
    :return: the image with missing values interpolated
    
    """
    if not np.isnan(vec).any():
        return vec
    mask = np.ma.masked_invalid(vec).mask
    h, w = vec.shape
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))


    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = vec[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]


    interp_values = interpolate.griddata(
        (known_x, known_y), 
        known_v, 
        (missing_x, missing_y),
        method=method, fill_value=fill_value
    )
    interp_image = vec.copy()
    interp_image[missing_y, missing_x] = interp_values
    return interp_image

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_boarders(vec: np.ndarray) -> np.ndarray:
    if not np.isnan(vec).any():
        return vec
    nans, x = nan_helper(vec[0,:])
    vec[0,nans]   = np.interp(x(nans), x(~nans), vec[0,:][~nans])    
    nans, x = nan_helper(vec[-1,:])
    vec[-1,nans] = np.interp(x(nans), x(~nans), vec[-1,:][~nans])
    nans, x = nan_helper(vec[:,0])
    vec[nans,0]   = np.interp(x(nans), x(~nans), vec[:,0][~nans])
    nans, x = nan_helper(vec[:,-1])
    vec[nans,-1] = np.interp(x(nans), x(~nans), vec[:,-1][~nans])
    
    return vec

def peak2peak_secondpeak(
    corr: torch.Tensor, imax: torch.Tensor, 
    wind: int=2) -> torch.Tensor:
    c, d, k = corr.shape
    cor = corr.view(c, -1)
    for i in range(-wind, wind+1):
        for j in range(-wind, wind+1):
            ids = imax + i + k * j
            ids[ids < 0] = 0
            ids[ids > k*d-1] = k*d - 1
            cor.scatter_(-1, ids, 0.0)
    second_max = cor.argmax(-1, keepdim=True)
    return second_max

def correlation_to_displacement(
    corr: torch.Tensor,
    n_rows, n_cols,
    validate: bool=True,
    val_ratio=1.2, 
    validation_window=3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_mask = None
    c, d, k = corr.shape
    eps = 1e-7
    corr += eps
    cor = corr.view(c, -1).type(torch.float64)
    m = corr.view(c, -1).argmax(-1, keepdim=True)

    left = m + 1
    right = m - 1
    top = m + k 
    bot = m - k
    left[left >= k*d - 1] = m[left >= k*d - 1]
    right[right <= 0] = m[right <= 0]
    top[top >= k*d - 1] = m[top >= k*d - 1]
    bot[bot <= 0] = m[bot <= 0]

    cm = torch.gather(cor, -1, m)
    cl = torch.gather(cor, -1, left)
    cr = torch.gather(cor, -1, right)
    ct = torch.gather(cor, -1, top)
    cb = torch.gather(cor, -1, bot)
    nom1 = torch.log(cr) - torch.log(cl) 
    den1 = 2 * (torch.log(cl) + torch.log(cr)) - 4 * torch.log(cm) 
    nom2 = torch.log(cb) - torch.log(ct) 
    den2 = 2 * (torch.log(cb) + torch.log(ct)) - 4 * torch.log(cm) 





    m2d = torch.cat((m // d, m % k), -1)
    v = m2d[:, 0][:, None] + nom2/den2
    u = m2d[:, 1][:, None] + nom1/den1
    # v[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = torch.nan
    # u[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = torch.nan
    if validate:
        # v[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = torch.nan
        # u[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = torch.nan
        m2 = peak2peak_secondpeak(corr, m, validation_window)
        validation_mask = (cm / torch.gather(cor, -1, m2)) < val_ratio
        validation_mask[(left >= k*d - 1) * (right <= 0) * (top >= k*d - 1) * (bot <= 0)] = True
        validation_mask = validation_mask.reshape(n_rows, n_cols).cpu().numpy()
        # u[validation_mask] = torch.nan
        # v[validation_mask] = torch.nan

    u = u.reshape(n_rows, n_cols).cpu().numpy()
    v = v.reshape(n_rows, n_cols).cpu().numpy()
     
    default_peak_position = np.floor(np.array(corr[0, :, :].shape)/2)
    v = v - default_peak_position[0]
    u = u - default_peak_position[1] 
    u = interpolate_boarders(u)
    v = interpolate_boarders(v)
    u = fastSubpixel.replace_nans(u) 
    v = fastSubpixel.replace_nans(v) 
    u = interpolate_nan(u)
    v = interpolate_nan(v)
    return u, v, validation_mask

def c_correlation_to_displacement(
    corr: torch.Tensor, 
    n_rows, n_cols, interp_nan=False) -> Tuple[np.ndarray]:
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 3D torch.Tesnsor [channels, :, :]
            contains output of the fft_correlate_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
    """
    # iterate through interrogation widows and search areas
    eps = 1e-7
    first_peak = find_first_peak_position(corr)
    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, :, :].shape)/2)
    corr += eps
    corr = corr.cpu().numpy()
    first_peak = first_peak.cpu().numpy()    

    temp = fastSubpixel.find_subpixel_position(corr, first_peak, n_rows, n_cols)
    peak = (np.array(temp).T - default_peak_position.T).T
    u, v = peak[0], peak[1]
    u, v = u.squeeze(), v.squeeze()
    # u = interpolate_boarders(u)
    # v = interpolate_boarders(v)
    # if interp_nan:
    #     u = interpolate_nan(u)
    #     v = interpolate_nan(v)
    return u, v


def get_field_shape(image_size, search_area_size, overlap) -> Tuple:
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape

    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two

    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    field_shape : three elements tuple
        the shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - search_area_size) // (
        search_area_size - overlap
    ) + 1
    return field_shape

def resize_iteration(arr: np.ndarray, shape: tuple=None):
    arr = cv2.resize(arr, shape, interpolation=cv2.INTER_LINEAR)
    # Depricated method, may use later
    # arr = tinterpolate(torch.from_numpy(arr[None, None, ...]), scale_factor=2, mode='bilinear', align_corners=True).numpy()
    return arr.squeeze()

def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size,
    overlap=0,
    validate: bool = True,
    validation_ratio:float = 1.2,
    device: torch.device=torch.device("cpu")
) -> Tuple[np.ndarray, ...]:
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. ZERO ORDER!!!

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.

    Parameters
    ----------
    frame_a : 2d torch.Tensor
        an two dimensions array of integers containing grey levels of
        the first frame.

    frame_b : 2d torch.Tensor
        an two dimensions array of integers containing grey levels of
        the second frame.

    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].

    """

    frame_a, frame_b = frame_a.to(device), frame_b.to(device)

    if overlap >= window_size:
        raise ValueError("Overlap has to be smaller than the window_size")

    if (window_size > frame_a.shape[-2]) or (window_size > frame_a.shape[-1]):
        raise ValueError("window size cannot be larger than the image")
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size=window_size, overlap=overlap)
    x, y = get_coordinates(frame_a.shape, window_size, overlap)
    aa = moving_window_array(frame_a, window_size, overlap)
    bb = moving_window_array(frame_b, window_size, overlap)
    # Normalize Intesity
    # aa = aa / torch.mean(aa, (-2,-1), dtype=torch.float32, keepdim=True)
    # bb = bb / torch.mean(bb, (-2,-1), dtype=torch.float32, keepdim=True)
    
    corr = correalte_fft(aa, bb)
    # Normalize correlation
    corr = corr - torch.amin(corr, (-2, -1), keepdim=True)
    u, v, validation_mask = correlation_to_displacement(corr, n_rows, n_cols, validate, val_ratio=validation_ratio)
    return u, v, x, y, validation_mask

def get_coordinates(image_size, search_area_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B

    overlap: int = 0 (default is no overlap)
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    x : 2d torch.tensor
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d torch.tensor
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[-1], dtype=np.int32) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[-2], dtype=np.int32) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    
    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    x += (
        image_size[-1]
        - 1
        - ((field_shape[-1] - 1) * (search_area_size - overlap) +
            (search_area_size - 1))
    ) // 2
    y += (
        image_size[-2] - 1
        - ((field_shape[-2] - 1) * (search_area_size - overlap) +
           (search_area_size - 1))
    ) // 2

    # the origin 0,0 is at top left
    # the units are pixels

    return np.meshgrid(x, y)

def piv_iteration_CWS(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor, 
    x0: np.ndarray,  
    y0: np.ndarray,  
    u0: np.ndarray,  
    v0: np.ndarray,
    validation_mask: np.ndarray,
    wind_size: int, 
    overlap: int,
    device: torch.device) -> tuple[np.ndarray, ...]:

    iter_proc = time()
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size=wind_size, overlap=overlap)
    x, y = get_coordinates(frame_a.shape, wind_size, overlap)
    spline_u = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], u0)
    spline_v = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], v0)
    u0 = spline_u(y[:,0], x[0,:])
    v0 = spline_v(y[:,0], x[0,:])
    if validation_mask is not None:
        spline_val = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], validation_mask)
        val = spline_val(y[:,0], x[0,:]) >= .5
        u0[val] = 0.0
        v0[val] = 0.0
    uflat = u0.flatten()
    vflat = v0.flatten()
    frame_a, frame_b = frame_a.to(device), frame_b.to(device)
    aa = moving_window_array(frame_a, wind_size, overlap)[:,None,...].float()
    bb = moving_window_array(frame_b, wind_size, overlap)[:,None,...].float()
    
    affine_transform = torch.tensor([[1., 0., 0.],
                                    [0., 1., 0.]]).to(device)
    affine_transform = affine_transform.repeat(aa.shape[0], 1, 1)
    affine_transform[:, 1, 2] = torch.from_numpy(-vflat/wind_size)
    affine_transform[:, 0, 2] = torch.from_numpy(-uflat/wind_size)
    grid = F.affine_grid(affine_transform, aa.size())
    aa = F.grid_sample(aa, grid, mode='bilinear',padding_mode="border")
    
    affine_transform[:, 1, 2] = torch.from_numpy(vflat/wind_size)
    affine_transform[:, 0, 2] = torch.from_numpy(uflat/wind_size)
    grid = F.affine_grid(affine_transform, bb.size())
    bb = F.grid_sample(bb, grid, mode='bilinear',padding_mode="border")

    # Normalize Intesity
    aa = aa / torch.mean(aa, (-2,-1), dtype=torch.float32, keepdim=True)
    bb = bb / torch.mean(bb, (-2,-1), dtype=torch.float32, keepdim=True)
    
    corr = correalte_fft(aa, bb)
    corr = corr - torch.amin(corr, (-2, -1), keepdim=True)
    du, dv, val = correlation_to_displacement(corr.squeeze(), n_rows, n_cols)

    v = v0 + dv
    u = u0 + du

    mask_u = (du > u0) * (np.rint(u0) > 0)
    mask_v = (dv > v0) * (np.rint(v0) > 0)
    if val is not None:
        mask_u[val] = True
        mask_v[val] = True

    v[mask_v] = v0[mask_v]
    u[mask_u] = u0[mask_u]
    print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
    return u, v, x, y, val




def piv_iteration_DWS(
    frame_a: torch.Tensor, 
    frame_b: torch.Tensor, 
    x0:       np.ndarray, 
    y0:       np.ndarray, 
    u0:      np.ndarray, 
    v0:      np.ndarray,
    validation_mask: np.ndarray, 
    wind_size: int,
    overlap: int, 
    device: torch.device)->tuple[np.ndarray, ...]:

    iter_proc = time()
    frame_a = frame_a.numpy()
    frame_b = frame_b.numpy()
    n_rows, n_cols = get_field_shape(frame_a.shape, wind_size, overlap)
    x, y = get_coordinates(frame_a.shape, wind_size, overlap)

    spline_u = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], u0)
    spline_v = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], v0)

    u0 = spline_u(y[:,0], x[0,:])
    v0 = spline_v(y[:,0], x[0,:])


    vin = v0/2
    uin = u0/2
    if validation_mask is not None:
        spline_val = interpolate.RectBivariateSpline(y0[:,0], x0[0,:], validation_mask)
        val = spline_val(y[:,0], x[0,:]) >= .5
        vin[val] = 0.0
        uin[val] = 0.0
    bb = torch.from_numpy(fastSubpixel.iter_displacement_DWS(frame_b, x.astype(np.int32), y.astype(np.int32), uin,  vin, wind_size))
    aa = torch.from_numpy(fastSubpixel.iter_displacement_DWS(frame_a, x.astype(np.int32), y.astype(np.int32), -uin, -vin, wind_size))

    aa, bb = aa.to(device), bb.to(device)
    # Normalize Intesity
    # aa = aa / torch.mean(aa, (-2,-1), dtype=torch.float32, keepdim=True)
    # bb = bb / torch.mean(bb, (-2,-1), dtype=torch.float32, keepdim=True)

    corr = correalte_fft(aa, bb)
    corr = corr - torch.amin(corr, (-2, -1), keepdim=True)

    du, dv, val = correlation_to_displacement(corr, n_rows, n_cols)

    v = 2*np.rint(vin) + dv
    u = 2*np.rint(uin) + du

    mask_u = (du > u0) * (np.rint(u0) > 0)
    mask_v = (dv > v0) * (np.rint(v0) > 0)
    if val is not None:
        mask_u[val] = True
        mask_v[val] = True

    v[mask_v] = v0[mask_v]
    u[mask_u] = u0[mask_u]
    print(f"Iteration finished in {(time() - iter_proc):.3f} sec", end=" ")
    return u, v, x, y, val

class IterModMap:
    functions = {
        "DWS": piv_iteration_DWS,
        "CWS": piv_iteration_CWS
    }

def calc_mean(v_list: list):
    np_list = np.stack(v_list, axis=0)
    return np.mean(np_list, axis=0).squeeze()

class OfflinePIV:
    def __init__(
        self, folder: str, 
        device: str,
        file_fmt: str, 
        wind_size: int, 
        overlap: int,
        iterations: int = 1,
        iter_mod: str="DWS",
        dt: int = 1,
        scale:float = 1.,
        iter_scale:float = 2.
                ) -> None:
        self._wind_size = wind_size
        self._overlap = overlap
        self._dt = dt
        self._iter = iterations
        self._iter_scale = iter_scale
        self._scale = scale
        
        self._device = DeviceMap.devicies[device]
        self._dataset = PIVDataset(folder, file_fmt, 
                       transform=ToTensor(dtype=torch.uint8)
                      )
        self._iter_function = IterModMap.functions[iter_mod]

    def __len__(self) -> int:
        return len(self._dataset)

    def __call__(self) -> Generator:
        loader = torch.utils.data.DataLoader(self._dataset, 
            batch_size=None, num_workers=0, pin_memory=True)

        end_time = time() 
        for a, b in loader:
             with torch.no_grad():

                print(f"Load time {(time() - end_time):.3f} sec", end=' ')
                start = time()
                u, v, x, y, val = extended_search_area_piv(a, b, window_size=self._wind_size, 
                                                overlap=self._overlap, device=self._device)
                


                wind_size = self._wind_size
                overlap = self._overlap
                for _ in range(self._iter-1):
                    wind_size = int(wind_size//self._iter_scale)
                    overlap = int(overlap//self._iter_scale)                    
                    u, v, x, y, val = self._iter_function(a, b, x, y, u, v, val, wind_size, overlap, self._device)

                if val is not None:
                    u[val] = np.nan
                    v[val] = np.nan
                    # u = interpolate_boarders(u)
                    # v = interpolate_boarders(v)
                    # u = fastSubpixel.replace_nans(u) 
                    # v = fastSubpixel.replace_nans(v) 
                    u = interpolate_nan(u)
                    v = interpolate_nan(v)

                u =  np.flip(u, axis=0)
                v = -np.flip(v, axis=0)

                yield x, y, u, v
                end_time = time()
                print(f"Batch finished in {(end_time - start):.3f} sec")


class OnlinePIV:
    def __init__(
        self, folder: str, 
        device: str,
        file_fmt: str, 
        wind_size: int, 
        overlap: int,
        iterations: int = 1,
        dt: int = 1,
        scale:float = 1.,
        resize: int = 2,
        iter_scale:float = 2.
                ) -> None:
        self._wind_size = wind_size
        self._overlap = overlap
        self._dt = dt
        self._iter = iterations
        self._iter_scale = iter_scale
        self._resize = resize
        self._scale = scale
        
        self._device = DeviceMap.devicies[device]