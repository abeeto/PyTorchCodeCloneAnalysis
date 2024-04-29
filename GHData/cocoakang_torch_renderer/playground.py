import os
import cv2
import numpy as np
import time
import torch
import open3d as o3d
from torch_render import Setup_Config

CONFIG_PATH="wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
setup = Setup_Config({"config_dir":CONFIG_PATH})

sub_plane = np.arange(0,16*32).reshape(16,32)
sub_plane = sub_plane.T

mask = -np.ones([192,256],np.int32)
mask[64:64+32,64*2:64*2+16] = sub_plane
cv2.imwrite("test_mask.exr",mask.astype(np.float32))

sub_light_pos = setup.get_sub_light_pos(mask)
print(sub_light_pos)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sub_light_pos)
o3d.io.write_point_cloud("sub_plane.ply", pcd)