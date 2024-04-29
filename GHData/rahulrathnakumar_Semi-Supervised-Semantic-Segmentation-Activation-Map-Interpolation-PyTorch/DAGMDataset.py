import torch
import torchvision
import numpy as np
import os
import random
import itertools

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms


class DAGMDataset(DataLoader):
    def __init__(self, root_dir, image_set, n_class, mode = None):
        self.n_class = n_class # Num_classes including background
        self.mode = mode
        self.root_dir = root_dir
        self.image_set = image_set
        self.images = []
        self.targets = []
        self.img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))
        for img_name in self.img_list:
            img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
            target_filename = os.path.join(root_dir, 'labels/{:s}'.format(img_name))
            if os.path.isfile(img_filename) and os.path.isfile(target_filename):
                self.images.append(img_filename)
                self.targets.append(target_filename)

    def read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            img_list.append(next_line.rstrip())
        return img_list

    def transforms(self, image, target = None):
        resize_image = transforms.Resize((224,224))
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        totensor = transforms.ToTensor()
        image = resize_image.__call__(image)
        if target is not None:
            resize_target = transforms.Resize((224,224), interpolation = Image.NEAREST)
            target = resize_target.__call__(target)
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            if target is not None:
                target = transforms.functional.hflip(target)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            if target is not None:
                target = transforms.functional.vflip(target)
        if random.random() > 0.5:
            angle,_,scale,shear = transforms.RandomAffine.get_params(degrees = [-5,5], 
            translate = None, scale_ranges = (1.0, 1.5), shears = (-2,2), 
            img_size = (200,200))
            image = transforms.functional.affine(image, angle = angle, 
            translate = [0,0], scale = scale, shear = shear)
            if target is not None:
                target = transforms.functional.affine(target, angle = angle, 
                translate = [0,0], scale = scale, shear = shear)            
        image = totensor(image)
        image = normalize(image)
        
        if target is not None:
            return image, target
        else:
            return image
        
    def one_hot(self,target):
        target = np.asarray(target)
        target = torch.from_numpy(target).long()
        h,w = target.size()
        label = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            label[c][target == c] = 1
        return label

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        labeled_image_list = self.read_image_list(os.path.join(self.root_dir, '{:s}.txt'.format('labeled')))
        if os.path.basename(self.targets[index]) in labeled_image_list or self.image_set =='val':
            target = Image.open(self.targets[index]).convert('L')
            image, target = self.transforms(image, target)
            label = self.one_hot(target)
            if self.mode == 'inference':
                return image, label, self.images[index]
            return image, label
        else:
            image = self.transforms(image)
            return image, -1*torch.ones(self.n_class, image.shape[1], image.shape[2])

    def __len__(self):
        return len(self.img_list)

class DataSampler(Sampler):
    def __init__(self, dataset, root_dir, image_set, labeled_batch_size, unlabeled_batch_size):
        self.dataset = dataset
        self.root_dir = root_dir
        self.image_set = image_set
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled = self.read_image_list(root_dir + 'labeled.txt')
        self.unlabeled = self.read_image_list(root_dir + 'unlabeled.txt')
        self.img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))

    def read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            img_list.append(next_line.rstrip())
        return img_list

    def sample_from_list(self, lst, chunk_size):
        return random.sample(lst, chunk_size)  

    def indexBatch(self, batch):
        indexed = []
        for sub_batch in batch:
            for name in sub_batch:
                indexed.append(self.img_list.index(name))
        return indexed

    def chunk(self, labeled, unlabeled, labeled_batch_size, unlabeled_batch_size):
        m = labeled_batch_size
        n = unlabeled_batch_size
        assert unlabeled_batch_size % 2 == 0, "Unlabeled batch size is not exactly divisible by 2."
        unlabeled_subbatch = [unlabeled[i:i+n] for i in range(0, len(unlabeled), n)]
        for j, minibatch  in enumerate(unlabeled_subbatch): 
            unlabeled_subbatch[j] = [minibatch[i: i + n//2] for i in range(0, n, n//2)]
        # Determine number of labeled chunks based on size of unlabeled_subbatch
        nb_labeled_chunks = len(unlabeled_subbatch)
        labeled_subbatch = [self.sample_from_list(labeled, m) for _ in range(nb_labeled_chunks)]
        labeled_subbatch = [[l] for l in labeled_subbatch] 
        combi = []
        for l, u in itertools.zip_longest(labeled_subbatch, unlabeled_subbatch):
            if l and u: combi.append(l + u)
            else: [combi.append(k) for k in [l,u] if k != None]
        return combi

    def __iter__(self):
        # First shuffle the labeled and unlabeled data
        random.shuffle(self.labeled)
        random.shuffle(self.unlabeled)
        # Combine the labeled and unlabeled data chunks into 1 batch
        combi = self.chunk(self.labeled, self.unlabeled, self.labeled_batch_size, self.unlabeled_batch_size)
        # Return the indices of the batch in the img_list.txt file
        combi_indices = list([self.indexBatch(i) for i in combi])
        return iter(combi_indices)
    def __len__(self):
        # print("NotImplemented.")
        # # return(len(self.labeled) + len(self.unlabeled))//self.batch_size
        return 0


