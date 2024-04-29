import os
from PIL import Image
import sys
from concurrent.futures import ProcessPoolExecutor
import psutil
import time
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt




def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, time2-time1))
        return ret
    return wrap



class MiniRGBD(object):
    class Folders:
        depth = 'depth'
        rgb = 'rgb'
        fg_mask = 'fg_mask'
        joints_2d = 'joints_2Ddep'
        joints_3d = 'joints_3D'
        smil_params_folder = 'smil_params'  # unused as of yet

    def __init__(self, dirnum):
        dir = dirnum[0]
        num = dirnum[1]

        # Save original file paths
        self._depth_file = os.path.join(dir, MiniRGBD.Folders.depth, 'syn_' + num + '_depth.png')
        self._rgb_file = os.path.join(dir, MiniRGBD.Folders.rgb, 'syn_' + num + '.png')
        self._fg_mask_file = os.path.join(dir, MiniRGBD.Folders.fg_mask, 'mask_' + num + '.png')
        self._joints_2d_file = os.path.join(dir, MiniRGBD.Folders.joints_2d, 'syn_joints_2Ddep_' + num + '.txt')
        self._joints_3d_file = os.path.join(dir, MiniRGBD.Folders.joints_3d, 'syn_joints_3D_' + num + '.txt')

        # Open images.
        depth = Image.open(self._depth_file)
        rgb = Image.open(self._rgb_file)
        fg = Image.open(self._fg_mask_file)
        self._bbox = fg.getbbox()

        # Store only the nonzero region.
        self._fg_mask = fg.crop(self._bbox)
        self._depth = Image.new(depth.mode, self._fg_mask.size)
        self._rgb = Image.new(rgb.mode, self._fg_mask.size)

        self._depth.paste(depth.crop(self._bbox), mask=self._fg_mask)
        self._rgb.paste(rgb.crop(self._bbox), mask=self._fg_mask)

        # Finally, load and store the joint values.
        def extract_joints(_joints_file):
            num_joints = 24  # Number of joints annotated in each frame.
            joints = []
            with open(_joints_file, 'r') as joints_file:
                for idx, line in enumerate(joints_file.readlines()):
                    x, y, z, joint_id = line.split()
                    x, y, z, joint_id = float(x), float(y), float(z), int(joint_id)
                    joints.append((x, y, z))
                    assert (joint_id == idx)  # Check that data format is always the same.
                assert (idx == num_joints)  # Make sure that all joints are present.
            return joints

        # Extract joints values
        self.joints_2d = extract_joints(self._joints_2d_file)
        self.joints_3d = extract_joints(self._joints_3d_file)

    @staticmethod
    def _thumbnail(img, thumbnail_size):
        # Since thumbnail preserves aspect ratio, it's size will be
        img = img.copy()
        img.thumbnail(thumbnail_size, Image.LANCZOS)
        padded = Image.new(img.mode, thumbnail_size)
        padded.paste(img,
                     ((padded.size[0] - img.size[0]) // 2,
                      (padded.size[1] - img.size[1]) // 2))
        return padded

    def depth_thumbnail(self, thumbnail_size):
        return MiniRGBD._thumbnail(self._depth, thumbnail_size)

    def rgb_thumbnail(self, thumbnail_size):
        return MiniRGBD._thumbnail(self._rgb, thumbnail_size)

    def point_cloud(self):
        z = np.asarray(Image.open(self._depth_file))

        # From depth_to_3D.py
        # camera calibration used for generation of depth
        fx = 588.67905803875317
        fy = 590.25690113005601
        cx = 322.22048191353628
        cy = 237.46785983766890

        # create tuple containing image indices
        indices = tuple(np.mgrid[:z.shape[0], :z.shape[1]].reshape((2, -1)))
        pts3D = np.zeros((indices[0].size, 3))
        pts3D[:, 2] = z[indices].ravel() / 1000.
        pts3D[:, 0] = (np.asarray(indices).T[:, 1] - cx) * pts3D[:, 2] / fx
        pts3D[:, 1] = (np.asarray(indices).T[:, 0] - cy) * pts3D[:, 2] / fy

        return pts3D


class MiniDDataset(torch.utils.data.Dataset):
    def __init__(self, mini_rgbd, thumbnail_size=(128, 128)):
        self.mini_rgbd = mini_rgbd
        self.thumbnail_size = thumbnail_size

    def __getitem__(self, index):
        rgbd = self.mini_rgbd[index]
        # Currently return depth image with 2d joints
        img = rgbd.depth_thumbnail(self.thumbnail_size)
        target = rgbd.joints_2d
        return img, target

    def __len__(self):
        return len(self.minirgbd)



def find_mini_rgbd(path):
    datafiles = []
    # Find all data files, given as (path, XXXXX):(str,str), where XXXXX is the number of the image, e.g. syn_XXXXX.png.
    with os.scandir(path) as it:
        for entry in it:
            # Get directories /XX/
            if entry.is_dir() and entry.name.isdigit():
                for file in os.listdir(os.path.join(entry, 'rgb')):
                    if file.startswith('syn_') and file.endswith(".png") and file[4:9].isdigit():
                        datafiles.append((entry.path, file[4:9]))
    return datafiles


def load_mini_rgbd(datafiles):
    # Since we max out the CPU, we lower our priority so the system does not become unresponsive.
    parent = psutil.Process()
    old_priority = parent.nice()
    parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # (Child processes inherit the niceness value)

    # Using multiple CPUs since each can handle a separate file. (Task is CPU-bound on SSD)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        datafiles = pool.map(MiniRGBD, datafiles, chunksize=64)
    parent.nice(old_priority)  # Restore process priority.
    sys.stdout.flush()  # TODO: Remove this
    return [data for data in datafiles]

import matplotlib.animation as animation

if __name__ == "__main__":
    data_files = find_mini_rgbd(os.path.join('RGBD', 'MINI-RGBD'))
    data_files = load_mini_rgbd(data_files[:2])
    # Now that we have the data (80 seconds load)
    fig, ax = plt.subplots()
    ims = []
    for (img, target) in MiniDDataset(data_files):
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000//30, blit=True,
                                    repeat_delay=0)
    plt.show()

    sys.exit()
    for file in os.listdir(os.path.join('RGBD', 'MINI-RGBD', '01', 'rgb')):
        assert (file.startswith('syn_') and file.endswith(".png"))  # Check right format.
        int(file[4:9])

        sys.exit()
        if file.endswith(".png"):
            rgb = Image.open(os.path.join(rgb_dir, file.replace('mask', 'syn')))
            depth = Image.open(os.path.join(depth_dir, file.replace('mask', 'syn').replace('.png', '_depth.png')))
            fg = Image.open(os.path.join(fg_mask_dir, file))
            # Extract foreground and crop to it.
            fgbox = fg.getbbox()
            cropped_depth = Image.composite(depth, Image.new(depth.mode, depth.size), fg).crop(fgbox)
            cropped_rgb = Image.composite(rgb, Image.new(rgb.mode, rgb.size), fg).crop(fgbox)

            depth_range = depth.getextrema()
            assert (depth_range[0] != 0)

            if not cropped_depth.getbbox() == cropped_rgb.getbbox():
                print(cropped_depth.getbbox() == cropped_rgb.getbbox())
                sys.exit()
            # Resize the image, preserving aspect ratio and pad it out, so it is the same as thumbnail_size
            # cropped.thumbnail(thumbnail_size, Image.LANCZOS)
            # padded = Image.new(cropped.mode, thumbnail_size)
            # padded.paste(cropped,
            #             (int((thumbnail_size[0] - cropped.size[0]) / 2),
            #              int((thumbnail_size[1] - cropped.size[1]) / 2)))

            # Make sure image is square (stackoverflow.com/questions/1386352/pil-thumbnail-and-end-up-with-a-square-image)
            # background.save("output.png")

            # rgb.crop(foreground.getbbox())
            sys.exit()

    #print([test._bbox for test in datafiles])

#with concurrent.futures.ProcessPoolExecutor() as executor:
#    datafiles = executor.starmap(MiniRGBD, datafiles)

# Store them as a dataset.
#data_files = [MiniRGBD(dir, num) for (dir, num) in datafiles]










#with open(path, 'rb') as f: