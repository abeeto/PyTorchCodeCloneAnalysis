from torchvision.io import read_image, write_png, write_jpeg
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import Resize
import torch.nn as nn
import torch
import sys
import time
import glob
import random
import numpy as np
from PIL import Image

from matplotlib import pyplot

device = "cuda" if torch.cuda.is_available() else "cpu"

images_folder = "C:/Users/justi/Desktop/MosaicWithPyTorch/AlleBilder/"
target_image = images_folder + "hochzytfotos_280821_270.JPG"

filenames = glob.glob(images_folder+"*.JPG")

target_tensor = read_image(target_image).to(device)#torch.from_numpy(np.array(to_pil_image(read_image(target_image), mode="YCbCr"))/256).transpose(0, 2).transpose(1, 2).to(device)#

target_subdivisions = 224 #number of "subpatches" that the target image will be divided into (changing this requires no re-preprocessing)
search_subdivisions = 2 #number of subdivisions of a single search patch
insertion_subdivisions = 160 #final resolution of the inserted mini-images
num_samples = 5 #the top-k similar samples from which one will be chosen with a probability distribution based on the normalized reciprocal colour difference

image_ratio = round(target_tensor.shape[2]/target_tensor.shape[1], 2)

search_preprocess = False
insertion_preprocess = True

def print_progress(current, max, start_time=0):
    percent=current/max
    passed_time=time.time()-start_time
    if start_time!=0:
        sys.stdout.write(f"\rProgress: {current}/{max} ({percent * 100:.1f}%) (Time: {passed_time:.0f}s / {passed_time / percent:.0f}s)")
    else:
        sys.stdout.write(f"\rProgress: {current}/{max} ({percent * 100:.1f}%)")
    sys.stdout.flush()

def compute_averages(img_path, subdivisions):
    img_tensor = read_image(img_path).float().to(device)#torch.from_numpy(np.array(to_pil_image(read_image(img_path), mode="YCbCr"))/256).transpose(0, 2).transpose(1, 2).float().to(device)

    #resize to better-divisible resulution
    resizer = Resize((3584, 5376), antialias=True)
    img_tensor = resizer(img_tensor)

    _, height, width = img_tensor.shape
    sub_height = int(height/subdivisions)
    sub_width = int(width/subdivisions)
    #if height>width: # apparently, this doesn't work when detecting vertical images. might need to find a different way to remove those in the end
    #    print("vertical image tossed")
    #    return "" # we don't want any vertical-format images

    avgpool = nn.AvgPool2d((sub_height, sub_width), stride=(sub_height, sub_width))#, stride=(sub_height, sub_width)).to(device)  # avgpool over every channel to get "subpatch" average

    avg_colors = avgpool(img_tensor).swapaxes(0, 1).swapaxes(1, 2).type(torch.uint8)

    return avg_colors

def generate_nearest_set(target_patch, average_matrix): # this function should compute the "distance" to the optimum for every image and return the ones that have the lowest distance
    average_matrix = average_matrix.view(average_matrix.shape[0], -1)
    target_patch = target_patch.unsqueeze(0).reshape(1, -1)

    assert average_matrix.shape[1]==target_patch.shape[1], f"Dimensionality mismatch: average_matrix.shape={average_matrix.shape}; target_patch.shape={target_patch.shape}"

    #compute similarity, based on formula from https://en.wikipedia.org/wiki/Color_difference

    target_patch = target_patch.reshape(target_patch.shape[0], 4, -1)
    average_matrix = average_matrix.reshape(average_matrix.shape[0], 4, -1)

    redmean = (average_matrix[:, :, 0]+target_patch[:, :, 0])/2
    deltared = (average_matrix[:, :, 0]-target_patch[:, :, 0])**2
    deltagreen = (average_matrix[:, :, 1]-target_patch[:, :, 1])**2
    deltablue = (average_matrix[:, :, 2]-target_patch[:, :, 2])**2

    redfactor = (2+redmean/256)
    greenfactor = 4
    bluefactor = (2+(255-redmean)/256)

    similarity_matrix = torch.sum(torch.sqrt(redfactor * deltared + greenfactor * deltagreen + bluefactor * deltablue), dim=1)

    #similarity_matrix = (((average_matrix-target_patch).float()/256)**2).sum(dim=1)
    similarity_values, top_indices = torch.topk(similarity_matrix, num_samples, largest=False)

    similarity_values=similarity_values.cpu()
    similarity_values = 1/similarity_values # the image with the smallest difference in colour should have the highest probability of being drawn
    top_indices = top_indices.cpu()

    similarity_values = similarity_values/torch.sum(similarity_values) #normalize draw probability
    choice=np.random.choice(np.arange(0, num_samples), p=similarity_values.numpy())

    top_indices = torch.tensor([top_indices[choice]])

    return top_indices

if search_preprocess:
    print(f"\nPreprocessing search patches...")
    average_matrix = torch.tensor([]).to(device)
    start_time=time.time()
    for i, image_path in enumerate(filenames):
        img_average = compute_averages(image_path, search_subdivisions)
        average_matrix=torch.cat((average_matrix, img_average.unsqueeze(0)), 0)
        print_progress(i+1, len(filenames), start_time)
    torch.save(average_matrix, f"average_matrix_cache_{search_subdivisions}.pt")
else:
    try:
        average_matrix = torch.load(f"average_matrix_cache_{search_subdivisions}.pt")
    except:
        print("No cached tensors found. Aborting...")
        exit(-1)

if insertion_preprocess:
    print(f"\nPreprocessing insertion patches...")
    insertion_resizer = Resize((insertion_subdivisions, int(insertion_subdivisions * image_ratio)), antialias=True)

    insertion_matrix = torch.tensor([]).to(device)
    start_time=time.time()
    for i, image_path in enumerate(filenames):
        resized_patch = insertion_resizer(read_image(image_path).to(device))
        resized_patch = resized_patch.swapaxes(0, 1).swapaxes(1, 2)
        insertion_matrix = torch.cat((insertion_matrix, resized_patch.unsqueeze(0)), 0)
        print_progress(i + 1, len(filenames), start_time)
    torch.save(insertion_matrix, f"insertion_matrix_cache_{insertion_subdivisions}.pt")
else:
    try:
        insertion_matrix = torch.load(f"insertion_matrix_cache_{insertion_subdivisions}.pt")
    except:
        print("No cached insertion tensors found. Aborting...")
        exit(-1)
#target_resizer = Resize((target_subdivisions, int(target_subdivisions*image_ratio)), antialias=True)
#target_downsampled = target_resizer(target_tensor)#compute_averages(target_image, target_subdivisions*search_subdivisions)
#target_downsampled = target_downsampled.swapaxes(0, 1).swapaxes(1, 2) #bring it into the correct form for image printing

target_downsampled = compute_averages(target_image, target_subdivisions)
print(f"\nFinding optimal patches to insert...")
#select good fit for the target image subpatch
nearest_indices = []
start_time=time.time()
for j in range(int(target_downsampled.shape[0]/search_subdivisions)):
    for i in range(int(target_downsampled.shape[1]/search_subdivisions)):
        nearest = generate_nearest_set(target_downsampled[j*search_subdivisions:(j+1)*search_subdivisions, i*search_subdivisions:(i+1)*search_subdivisions, :], average_matrix)
        nearest_indices.append(nearest[0].item())
        if (i+j*target_subdivisions+1)%100==0:
            print_progress(i+j*target_subdivisions+1, int(target_downsampled.shape[0]/search_subdivisions*target_downsampled.shape[1]/search_subdivisions), start_time)

x=1

output_tensor = torch.zeros([int(target_subdivisions/search_subdivisions)*insertion_subdivisions, int(target_subdivisions/search_subdivisions*image_ratio)*insertion_subdivisions, 3]).int()

insertion_height = insertion_subdivisions
insertion_width = int(image_ratio * insertion_subdivisions)

print(f"\nInsertion process...")
start_time=time.time()
for j in range(int(target_downsampled.shape[0]/search_subdivisions)):
    for i in range(int(target_downsampled.shape[1]/search_subdivisions)):
        #output_tensor[j*search_subdivisions:(j+1)*search_subdivisions, i*int(image_ratio*search_subdivisions):(i+1)*int(image_ratio*search_subdivisions)] = average_matrix[nearest_indices[i+j*target_subdivisions]]
        #try:
        output_tensor[j*insertion_height:(j+1)*insertion_height, i*insertion_width:(i+1)*insertion_width, :] = insertion_matrix[nearest_indices[i+j*int(target_downsampled.shape[0]/search_subdivisions)]]
        #except:
        #    print("Error occured...")
        print_progress(i+j*int(target_downsampled.shape[0]/search_subdivisions)+1, int(target_downsampled.shape[0]/search_subdivisions)**2, start_time)

output_tensor = output_tensor.swapaxes(2, 1).swapaxes(1, 0).type(torch.uint8)
#write_png(output_tensor, f"{time.time()}_mosaic_png.png")
write_jpeg(output_tensor, f"{time.time()}_mosaic_jpeg.JPG", quality=80)
#pyplot.imshow(output_tensor)
#pyplot.show()
x=1