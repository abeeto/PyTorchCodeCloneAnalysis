import numpy as np

def psnr(img1, img2, PIXEL_MAX):
    mse = np.mean( (img1.astype(np.float32) - img2.astype(np.float32)) ** 2 )
    if mse == 0:
        return np.inf
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))