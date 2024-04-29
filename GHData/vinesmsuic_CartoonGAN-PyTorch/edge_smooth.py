import cv2
import numpy as np
import config
import os
from tqdm import tqdm


# Paper author used MedianBlur instead of Gaussian blur: https://github.com/FlyingGoblin/CartoonGAN/issues/11
def edge_smooth(image, FORMAT_BGR = True):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    
    #image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    pad_img = np.pad(image, ((2,2), (2,2), (0,0)), mode='reflect')

    if(FORMAT_BGR):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect edge pixels using a standard Canny edge detector
    edges = cv2.Canny(gray_img, 100, 200)
    # dilate the edge regions
    dilation = cv2.dilate(edges, kernel)

    # apply a Gaussian smoothing in the dilated edge regions
    gauss_img = np.copy(image)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    return gauss_img

def align_image_pair(cartoon, edge_smoothed_cartoon):
    # Concatenate 2 images
    aligned = np.concatenate((cartoon, edge_smoothed_cartoon), 1)
    return aligned

def produce_edge_dataset(cartoon_folder = config.TRAIN_CARTOON_DIR, cartoon_edge_folder = config.TRAIN_CARTOON_EDGE_DIR):

    print("="*80)
    print(str(os.path.basename(__file__)) + ": Cartoon Folder = "  + str(cartoon_folder))
    print(str(os.path.basename(__file__)) + ": Cartoon-Edge paired Folder = " + str(cartoon_edge_folder))
    print(str(os.path.basename(__file__)) + ": Start edge-smoothing.")

    cartoon_files = os.listdir(cartoon_folder)

    if not os.path.isdir(cartoon_edge_folder):
        os.makedirs(cartoon_edge_folder)

    count = 0
    
    for cartoon_file in tqdm(cartoon_files):
        bgr_cartoon = cv2.imread(os.path.join(cartoon_folder, cartoon_file))
        edge_cartoon = edge_smooth(bgr_cartoon, FORMAT_BGR=True)
        paired_image = align_image_pair(bgr_cartoon, edge_cartoon)
        count += 1
        cv2.imwrite(os.path.join(cartoon_edge_folder, str(count) + '.png'), paired_image)
    
    print(str(os.path.basename(__file__)) + ": Finished edge-smoothing")
    print(str(os.path.basename(__file__)) + ": Converted total of " + str(count) + " images.")
    print("="*80)
    

if __name__ == "__main__":
    
    produce_edge_dataset()
    