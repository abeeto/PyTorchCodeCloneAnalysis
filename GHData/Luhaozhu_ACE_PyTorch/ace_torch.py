from multiprocessing import dummy as multiprocessing
import sys
import os
import numpy as np
from PIL import Image
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import torch
import torch.nn as nn
from torchvision import transforms

from utils import save_images


# get feature map and grad information
grad_blobs = []
def hook_grad(module, grad_input, grad_output):
    grad_blobs.append(grad_output[0].data.cpu())

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu())


class ConceptDiscovery:
  def __init__(self,
              model, # 输入的神经网络模型
              img_loader, # 导入图片的数据类型
              target_class, # 解释的目标类别
              bottlenecks, # 目标要提取的神经网络层数 例如InceptionV3用的mix4c
              device,
              img_size=(224,224),
              # random_concept, # 设置的随机概念
              # activation_dir, # 激活值保存的路径
              # cav_dir,  # 保存cav的路径
              num_random_exp=2, # 在计算cav时使用多少次随机试验
              channel_mean=True, 
              max_imgs=40, # 最多拿多少个概念
              min_imgs=20, # 最少拿多少个概念
              num_discovery_imgs=40,  # 在概念探索的时候使用多少张图像
              num_workers=20,  # 并行运行
              average_image_value=117):
    self.model = model
    self.img_loader = img_loader
    self.target_class = target_class
    self.num_random_exp = num_random_exp
    self.img_size = img_size
    self.device = device
    # if isinstance(bottlenecks, str):
    #   bottlenecks = [bottlenecks]
    self.bottlenecks = bottlenecks
    # self.source_dir = source_dir
    # self.activation_dir = activation_dir
    # self.cav_dir = cav_dir
    # self.channel_mean = channel_mean
    # self.random_concept = random_concept
    # self.image_shape = model.get_image_shape()[:2]  # 获得图像的宽和高
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.num_workers = num_workers
    self.average_image_value = average_image_value
    self.channel_mean = channel_mean


  def create_patches(self,img_dir,method='slic',discovery_images=None,param_dict=None,save_img=True):
    if param_dict is None:
       param_dict = {}
    dataset, image_numbers, patches = [], [], []
    for image,target in self.img_loader:  # 迭代整个dataloader

      if discovery_images is None:
        raw_imgs = image.permute(0,2,3,1).cpu().numpy()
        self.image = image.to(self.device)  # 图像的张量保存形式
        self.discovery_images = raw_imgs
          
      else:
        self.discovery_images = discovery_images
      
      if save_img:
        save_images(img_dir,(self.discovery_images * 255).astype(np.uint8))  # 存储这个batch的每一张图片

      if self.num_workers:
        pool = multiprocessing.Pool(self.num_workers) # 并行运算
        # outputs: 获得该图像分割得到的所有的superpixel以及对应于原图的patch
        outputs = pool.map(
            lambda img: self._return_superpixels(img, method, param_dict), # 对图像进行超像素分割
            self.discovery_images)
        for fn, sp_outputs in enumerate(outputs):
          image_superpixels, image_patches = sp_outputs
          for superpixel, patch in zip(image_superpixels, image_patches):
            dataset.append(superpixel)  # 将超像素进行保存
            patches.append(patch)       # 将patch进行保存
            image_numbers.append(fn)
      else:
        for fn, img in enumerate(self.discovery_images):
          image_superpixels, image_patches = self._return_superpixels(  # 保存所有图像分割出来的所有pixels和patches
              img, method, param_dict)
          for superpixel, patch in zip(image_superpixels, image_patches):
            dataset.append(superpixel)  # 用来保存superpixel
            patches.append(patch)    # 用来保存对应的patch
            image_numbers.append(fn)  # 用来保存图像的索引号（因为一张图像对应了多个superpixel和patch，所以要保存图像的索引号）
    self.dataset, self.image_numbers, self.patches =\
          np.array(dataset).astype(np.float32), np.array(image_numbers), np.array(patches)

  def _return_superpixels(self, img, method='slic',param_dict=None):
    """Returns all patches for one image.

    Given an image, calculates superpixels for each of the parameter lists in
    param_dict and returns a set of unique superpixels by
    removing duplicates. If two patches have Jaccard similarity more than 0.5,
    they are concidered duplicates.

    Args:
      img: The input image
      method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    Raises:
      ValueError: if the segementation method is invaled.
    """
    if param_dict is None:
      param_dict = {}
    # pop(key[,default])
    # 如果没有key 就返回default值
    if method == 'slic':
      n_segments = param_dict.pop('n_segments', [15, 50, 80])
      n_params = len(n_segments) # 3
      compactnesses = param_dict.pop('compactness', [20] * n_params)
      sigmas = param_dict.pop('sigma', [1.] * n_params)
    elif method == 'watershed':
      markers = param_dict.pop('marker', [15, 50, 80])
      n_params = len(markers)
      compactnesses = param_dict.pop('compactness', [0.] * n_params)
    elif method == 'quickshift':
      max_dists = param_dict.pop('max_dist', [20, 15, 10])
      n_params = len(max_dists)
      ratios = param_dict.pop('ratio', [1.0] * n_params)
      kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
    elif method == 'felzenszwalb':
      scales = param_dict.pop('scale', [1200, 500, 250])
      n_params = len(scales)
      sigmas = param_dict.pop('sigma', [0.8] * n_params)
      min_sizes = param_dict.pop('min_size', [20] * n_params)
    else:
      raise ValueError('Invalid superpixel method!')
    unique_masks = []
    for i in range(n_params): # 对每一个像素级别进行分割
      param_masks = []
      if method == 'slic':
        segments = segmentation.slic(
            img, n_segments=n_segments[i], compactness=compactnesses[i],
            sigma=sigmas[i])
      elif method == 'watershed':
        segments = segmentation.watershed(
            img, markers=markers[i], compactness=compactnesses[i])
      elif method == 'quickshift':
        segments = segmentation.quickshift(
            img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
            ratio=ratios[i])
      elif method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
      for s in range(segments.max()):  # 遍历所有的分割区域
        # 对于分割区域而言，得到的是二值图像，一个像素值上面的所有点对应的是一个分割区域
        mask = (segments == s).astype(float)
        if np.mean(mask) > 0.001:
          unique = True
          for seen_mask in unique_masks:  # 如果两个mask相似程度过大，就把现在这个mask给去掉
            jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
            if jaccard > 0.5:
              unique = False
              break
          if unique:
            param_masks.append(mask)
      unique_masks.extend(param_masks)
    superpixels, patches = [], []
    while unique_masks:  # unique_masks是一个独一无二的分割区域
      # unique_masks.pop() 获取一个独一无二的mask

      # superpixel: 从图像中分割出来的那个区域然后resize成(224,224,3)的大小
      # patch: 分割区域对应于图像的区域，用mask取出来的roi
      superpixel, patch = self._extract_patch(img, unique_masks.pop())  
      superpixels.append(superpixel)
      patches.append(patch)
    return superpixels, patches
  
  def _extract_patch(self, image, mask):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    mask_expanded = np.expand_dims(mask, -1)  # 把mask变成(224,224,1)的
    patch = (mask_expanded * image + (
        1 - mask_expanded) * float(self.average_image_value) / 255)  # 取出图像中对应的分割区域
    ones = np.where(mask == 1)  # 获得分割区域对应的像素位置
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8)) # 取出对应的分割区域并建立一个patch
    image_resized = np.array(image.resize(self.img_size,
                                          Image.BICUBIC)).astype(float) / 255
    return image_resized, patch
  
  def _patch_activations(self, imgs, batch_size=256,channel_mean=None):
    """Returns activations of a list of imgs.
      获取在给定特征层的输出值，输出的是一个特征图像
    Args:
      imgs: List/array of images to calculate the activations of  # [dataset_length,3,224,224]
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """
    if channel_mean is None:
      channel_mean = self.channel_mean
    
    img_features = {feature_name:[] for feature_name in self.bottlenecks}
    del features_blobs[:]
    for i in range(int(imgs.shape[0] / batch_size) + 1):

      model_output = self.model(imgs[i * batch_size:(i + 1) * batch_size])
      
      for j,feature in enumerate(features_blobs):
        if channel_mean and len(feature.shape) > 3:
          feature = torch.mean(feature, dim=(2, 3))
        else:
          feature = feature.reshape(feature.shape[0], -1)  # [data_size,W*H*C]

        img_features[self.bottlenecks[j]] += [feature]  # return a numpy value
      
      del features_blobs[:]
    
    for key,value in img_features.items():
      img_features[key] = torch.cat(value).numpy()    # [dataset_length,512]
      # print(img_features[key].shape)
    return img_features  # return a dict

  def _patch_gradients(self, imgs, batch_size=256,num_classes=1000):
    """Returns activation gradients of a list of imgs.
      获取在给定特征层的输出值，输出的是一个特征图像
    Args:
      imgs: List/array of images to calculate the activations of  # [dataset_length,3,224,224]
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """

    
    img_grads = {feature_name:[] for feature_name in self.bottlenecks}
    for i in range(int(imgs.shape[0] / batch_size) + 1):

      input = imgs[i * batch_size:(i + 1) * batch_size].to(self.device)
      input.requires_grad_(True)
      model_output = self.model(input).to(self.device)
      self.model.zero_grad()
      model_output.backward(torch.ones(input.shape[0],num_classes).to(self.device))
      
      for j,grad in enumerate(grad_blobs):

        img_grads[self.bottlenecks[j]] += [grad]  # return a numpy value
      
      del grad_blobs[:]
    
    for key,value in img_grads.items():
      img_grads[key] = torch.cat(value).numpy()    # [dataset_length,512]
      # print(img_grads[key].shape)
    return img_grads  # return a dict
  
  def _cluster(self, acts, method='KM', param_dict=None):
    """Runs unsupervised clustering algorithm on concept actiavtations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer. [concept_lens,feature_nums]
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
    if param_dict is None:
      param_dict = {}
    centers = None
    if method == 'KM': # 使用 KMeans进行聚类
      n_clusters = param_dict.pop('n_clusters', 25) 
      km = cluster.KMeans(n_clusters)
      d = km.fit(acts)
      # d.labels_ 返回每一个概念对应的聚类类别
      centers = km.cluster_centers_  # 返回聚类中心
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)  # 计算每张图像到每个聚类中心之间的距离
      # asg:最小距离对应的图像是哪个 cost:最小距离是多大
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'AP':
      damping = param_dict.pop('damping', 0.5)
      ca = cluster.AffinityPropagation(damping)
      ca.fit(acts)
      centers = ca.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'MS':
      ms = cluster.MeanShift(n_jobs=self.num_workers)
      asg = ms.fit_predict(acts)
    elif method == 'SC':
      n_clusters = param_dict.pop('n_clusters', 25)
      sc = cluster.SpectralClustering(
          n_clusters=n_clusters, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    elif method == 'DB':
      eps = param_dict.pop('eps', 0.5)
      min_samples = param_dict.pop('min_samples', 20)
      sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    else:
      raise ValueError('Invalid Clustering Method!')
    if centers is None:  ## If clustering returned cluster centers, use medoids
      centers = np.zeros((asg.max() + 1, acts.shape[1]))
      cost = np.zeros(len(acts))
      for cluster_label in range(asg.max() + 1):
        cluster_idxs = np.where(asg == cluster_label)[0]
        cluster_points = acts[cluster_idxs]
        pw_distances = metrics.euclidean_distances(cluster_points)
        centers[cluster_label] = cluster_points[np.argmin(
            np.sum(pw_distances, -1))]
        cost[cluster_idxs] = np.linalg.norm(
            acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
            ord=2,
            axis=-1)
    # asg: 每一个概念所对应的聚类类别
    # cost 每一个概念对应聚类类别到该聚类中心之间的距离
    # center: 聚类中心
    return asg, cost, centers
  
  def discover_concepts(self,
                        method='KM',
                        param_dicts=None):
    """Discovers the frequent occurring concepts in the target class.

      Calculates self.dic, a dicationary containing all the informations of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept, 'patches': original patches of the
      concepts, 'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
    """
    if param_dicts is None:
      param_dicts = {}
    if set(param_dicts.keys()) != set(self.bottlenecks):
      param_dicts = {bn: param_dicts for bn in self.bottlenecks}
    self.dic = {}  ## The main dictionary of the ConceptDiscovery class.
    self.dataset_tensor = torch.tensor(self.dataset).permute(0,3,1,2).to(self.device)  # 把dataset写入到GPU中
    bn_mean_activations = self._patch_activations(self.dataset_tensor)  # 获取输入的概念图像(resize后的superpixel)在给定特征层的输出值
    bn_activations = self._patch_activations(self.dataset_tensor,channel_mean=False)
    bn_gradients = self._patch_gradients(self.dataset_tensor)
    for bn in self.bottlenecks:  # 遍历每一个特征层
      bn_dic = {}
      bn_dic['label'], bn_dic['cost'], centers = self._cluster(  # 进行聚类，将其聚类为n个类别，默认为25
          bn_mean_activations[bn], method, param_dicts[bn])
      concept_number, bn_dic['concepts'] = 0, []
      for i in range(bn_dic['label'].max() + 1): # for i in range(n_clusters):
        # 遍历每一个聚类类别
        label_idxs = np.where(bn_dic['label'] == i)[0] # 找到聚类类别为i的对应索引下标
        if len(label_idxs) > self.min_imgs:  # 如果该类别的聚类出的概念图片大于最小要求的图片
          concept_costs = bn_dic['cost'][label_idxs]  # 获得该聚类类别的图片到对应聚类中心之间的距离
          concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]] # 选出距离聚类类别中心最近的前n张概念图片，按距离从小到大进行排序
          concept_image_numbers = set(self.image_numbers[label_idxs]) # 找出这类概念在哪几张图像中出现
          discovery_size = len(self.discovery_images)  # 需要探索的图像总体个数是多少

          # highly_common_concept:（这里是为了验证这个概念在图像中出现的频率是否广泛，有可能一个图像有很多个相似的聚类得到的一组概念，
          # 但是这个概念只在几张图片中出现，这就是不common的）
          highly_common_concept = len(
              concept_image_numbers) > 0.5 * len(label_idxs)  # 如果有这些概念的图像个数 > 该类别概念全部个数 * 0.5 则为经常出现的概念
          mildly_common_concept = len(
              concept_image_numbers) > 0.25 * len(label_idxs) # 如果有这些概念的图像个数 > 该类别概念全部个数 * 0.25 则为中等出现的概念
          mildly_populated_concept = len(
              concept_image_numbers) > 0.25 * discovery_size  # 如果有这些概念的图像个数 > 全部探索的图像个数 * 0.25 则为中等流行的概念
          cond2 = mildly_populated_concept and mildly_common_concept
          non_common_concept = len(                    # 如果有这些概念的图像个数 > 该类别概念全部个数 * 0.25 则为不经常出现的概念
              concept_image_numbers) > 0.1 * len(label_idxs)
          highly_populated_concept = len(              # 如果有这些概念的图像个数 > 全部探索的图像个数 * 0.5 则为非常流行的概念（说明很多图像都有这个概念）
              concept_image_numbers) > 0.5 * discovery_size
          cond3 = non_common_concept and highly_populated_concept
          if highly_common_concept or cond2 or cond3:  # 如果满足这三个条件之一，则将该聚类类别保存为一个概念；反之则不保存
            concept_number += 1
            concept = '{}_concept{}'.format(self.target_class, concept_number)
            bn_dic['concepts'].append(concept)
            bn_dic[concept] = {
                'gradients':bn_gradients[bn][concept_idxs],
                'activations':bn_activations[bn][concept_idxs],
                'images': self.dataset[concept_idxs],  # 保存这个概念类别最相关的40个概念，以superpixel的形式保存
                'patches': self.patches[concept_idxs], # 保存这个概念类别最相关的40个概念，以patch的形式保存
                'image_numbers': self.image_numbers[concept_idxs] # 保存这个概念类别对应的是哪几张图像
            }
            bn_dic[concept + '_center'] = centers[i]  # 保存这个概念类别的聚类中心
      bn_dic.pop('label', None)
      bn_dic.pop('cost', None)
      self.dic[bn] = bn_dic
    self.dic['discovery_images'] = self.discovery_images
  
  def save_concepts(self, concepts_dir):
    """Saves discovered concept's images or patches.

    Args:
      cd: The ConceptDiscovery instance the concepts of which we want to save
      concepts_dir: The directory to save the concept images
    """
    for bn in self.bottlenecks:  # 遍历每一个聚类的特征层
      for concept in self.dic[bn]['concepts']:  # 获取对应的概念
        patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')  #
        images_dir = os.path.join(concepts_dir, bn + '_' + concept)
        patches = (np.clip(self.dic[bn][concept]['patches'], 0, 1) * 256).astype(  # 分割对应在图像上的区域
            np.uint8) 
        images = (np.clip(self.dic[bn][concept]['images'], 0, 1) * 256).astype(  # 对应的superpixel
            np.uint8)
        os.makedirs(patches_dir)  
        os.makedirs(images_dir)
        image_numbers = self.dic[bn][concept]['image_numbers']  # 保存该概念对应于哪几个图像
        image_addresses, patch_addresses = [], []
        for i in range(len(images)):
          image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format( 
              i + 1, image_numbers[i])
          patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))  # 保存对应的patch概念
          image_addresses.append(os.path.join(images_dir, image_name + '.png'))   # 保存对应的superpixel
        save_images(patch_addresses, patches)
        save_images(image_addresses, images)
      
    # save self.dic
    dict_path_name = os.path.join(concepts_dir,'concept_info.npy')
    np.save(dict_path_name, self.dic)
  
  def get_random_activation(self,random_path,random_save_dir,save=True):

    concept_num = len(self.dic[self.bottlenecks[0]]['concepts'])
    print(f"number of concepts:{concept_num}")
    transform_fn = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),  # 变成Tensor格式，归一化到[0,1区间]
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化均值和方差，与ImageNet的均值和方差一致
            ])
    random_activations = {}
    for i in range(concept_num):
      random_name = f"random500_{i}"
      random_img_path = os.path.join(random_path,random_name)
      img_list = []
      for img_path in os.listdir(random_img_path):
        img = Image.open(os.path.join(random_img_path,img_path)).convert('RGB')
        img = transform_fn(img).unsqueeze(0)
        img_list.append(img)
      img_tensor = torch.cat(img_list).to(self.device)   # 获取所有的random image [500,3,224,224]

      random_features = self._patch_activations(img_tensor,channel_mean=False)
      random_activations[random_name] = random_features
    
    if save:
      dict_path_name = os.path.join(random_save_dir,'random_info.npy')
      np.save(dict_path_name, random_activations)
    
    return random_activations

  def construct_concept(self,img_dir,concept_dir,random_path,segment_method='slic',cluster_method='KM',
                            param_dict=None,save_concept=True,save_img=True,save_dict=True,save_random=True):

    self.create_patches(img_dir,method=segment_method,param_dict=param_dict,save_img=save_img)  # 建立patches
    self.discover_concepts(method=cluster_method, param_dicts=param_dict) # 对概念进行聚类并保存

    random_activations = self.get_random_activation(random_path=random_path,random_save_dir=concept_dir,save=save_random)

    if save_concept:
      self.save_concepts(concepts_dir=concept_dir)

    del self.dataset
    del self.image_numbers
    del self.patches
    del self.dataset_tensor
    del self.dic