from pathlib import Path
import numpy as np
import cv2
import argparse
import tqdm


def make_edge_smooth(dataset_name: str = "Shinkai", img_size: int = 256):

    file_list = Path(f'./dataset/{dataset_name}/style').resolve()
    save_dir = Path(f'./dataset/{dataset_name}/smooth').resolve()

    if not file_list.exists():
        raise FileNotFoundError()

    if not save_dir.exists():
        save_dir.mkdir()

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list.glob('**/*')):

        file_name = f.name

        bgr_img = cv2.imread(str(f))
        gray_img = cv2.imread(str(f), 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


def main():
    if __name__ == '__main__':
        make_edge_smooth('Shinkai', 256)
