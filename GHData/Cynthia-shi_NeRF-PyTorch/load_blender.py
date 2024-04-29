import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


# 通过 load_blender_data() 函数得到指定文件夹下的所有图像、pose、测试渲染的pose、宽高焦距以及分割数组。
def load_blender_data(basedir, half_res=False, testskip=1):
    """
	:param basedir: 数据文件夹路径
    :param half_res: 是否对图像进行半裁剪
    :param testskip: 挑选测试数据集的跳跃步长
    :return: 
	"""
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        # 分别加载三个 .json 文件，保存到字典中
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]   # 加载 .json 文件中的内容
        imgs = []
        poses = []
        # 如果是 train 文件夹，连续读取图像数据
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            # 以指定步长读取列表中的字典
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))  # 将读取到的图像文件添加到图像列表中
            poses.append(np.array(frame['transform_matrix'])) # 将读取到的转置矩阵添加到转置矩阵列表中
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA) 此时 图像数组是一个 [n, H, W, 4] 的数组
        poses = np.array(poses).astype(np.float32)    # [n, 4, 4]
        counts.append(counts[-1] + imgs.shape[0])  # 用于标定到train、test、val

        all_imgs.append(imgs)    # 包含了 train、test、val 的图像的列表
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)] 
    
    imgs = np.concatenate(all_imgs, 0)  # 把列表聚合成为一个数组 [N, H, W, 4]
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)  # 计算焦距
    
    # 制作用于测试训练效果的 渲染pose
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 为了节省内存开销可以选择只加载一半的数据
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


