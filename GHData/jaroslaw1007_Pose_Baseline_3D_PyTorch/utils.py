import numpy as np
import copy
import cameras

from config import *
from cameras import *

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def normalization_stats(complete_data, dim):
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    
    dimensions_to_ignore = []
    
    if dim == 2:
        dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dimensions_to_use)
    else:
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.delete(dimensions_to_use, 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3, dimensions_to_use * 3 + 1, dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3), dimensions_to_use)
        
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def normalize_data(data, data_mean, data_std, dimensions_to_use):
    normalized_data = {}
    
    for key in data.keys():
        # key: (subject, action, name) -> [whole sequence, dim]
        data[key] = data[key][:, dimensions_to_use]
        mu = data_mean[dimensions_to_use]
        stddev = data_std[dimensions_to_use]
        normalized_data[key] = np.divide((data[key] - mu), stddev)
        
    return normalized_data

def unNormalize_data(normalized_data, data_mean, data_std, dimensions_to_use):
    T = normalized_data.shape[0]
    D = data_mean.shape[0]
    
    origin_data = np.zeros((T, D), dtype=np.float32)
    origin_data[:, dimensions_to_use] = normalized_data
    
    std_matrix = data_std.reshape((1, D))
    std_matrix = np.repeat(std_matrix, T, axis=0)
    mean_matrix = data_mean.reshape((1, D))
    mean_matrix = np.repeat(mean_matrix, T, axis=0)
    
    origin_data = np.multiply(origin_data, std_matrix) + mean_matrix
    
    return origin_data

def postprocess_3d(poses_set):
    root_coords = {}
    for key in poses_set.keys():
        root_coords[key] = copy.deepcopy(poses_set[key][:, :3])
        
        poses = poses_set[key]
        poses = poses - np.tile(poses[:, :3], [1, 32])
        poses_set[key] = poses
    
    return poses_set, root_coords 
        
def project_to_cameras(poses_set, cams, ncams=4):
    t2d = {}
    
    for t3d_key in sorted(poses_set.keys()):
        subject, action, seq_name = t3d_key
        t3d = poses_set[t3d_key]
        
        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subject, cam + 1)]
            points_2d, _, _, _, _ = cameras.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)
            points_2d = np.reshape(points_2d, [-1, 64])
            sname = seq_name[:-3] + '.' + name + '.h5'
            t2d[(subject, action, sname)] = points_2d
            
    return t2d

def transform_world_to_camera(poses_set, cams, ncams=4):
    t3d_camera = {}
    
    for t3d_key in sorted(poses_set.keys()):
        subject, action, seq_name = t3d_key
        t3d_world = poses_set[t3d_key]
        
        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subject, cam + 1)]
            camera_coord = world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES) * 3])
            
            sname = seq_name[:-3] + '.' + name + '.h5'
            t3d_camera[(subject, action, sname)] = camera_coord
            
    return t3d_camera