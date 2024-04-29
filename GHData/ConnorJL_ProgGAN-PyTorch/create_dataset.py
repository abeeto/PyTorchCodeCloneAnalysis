import os

from scipy.misc import imread, imresize, imsave

data_folder = "img_align_celeba"
output_folder = "celeba"

l = []
for root, dirs, files in os.walk(data_folder):
    for f in files:
        l.append(os.path.join(data_folder, f))

sizes = [4, 8, 16, 32, 64, 128, 256]

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for i in sizes:
    d = os.path.join(output_folder, str(i))
    if not os.path.exists(d):
        os.mkdir(d)
    d = os.path.join(d, "imgs") # For some strange reason, torch doesn't like images being in the top level folder
    if not os.path.exists(d):
        os.mkdir(d)

i = 0
out = []
for n in l:
    name = n.split("/")[-1]
    new_name = name.split(".")[0]+".png"
    try:
        im = imread(os.path.join(data_folder, name), mode="RGB")
        for s in sizes:
            if not os.path.exists(os.path.join(output_folder, str(s), "imgs", new_name)):
                tmp = imresize(im, (s, s))
                imsave(os.path.join(output_folder, str(s), "imgs", new_name), tmp)
        out.append(new_name)
    except:
        for s in sizes:
            f = os.path.join(output_folder, str(s), "imgs", new_name)
            if os.path.exists(f):
                os.remove(f) # Just in case something weird happens so that no faulty images are left

