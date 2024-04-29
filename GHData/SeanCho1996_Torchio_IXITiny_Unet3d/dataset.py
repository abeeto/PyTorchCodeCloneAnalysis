from glob import glob
import os
import torch
import torchio as tio
import json

from torchio import ScalarImage, LabelMap


class Datasets3D():
    def __init__(self, dataset_dir, training_split_ratio=0.9):
        images_dir = os.path.join(dataset_dir, 'image')
        labels_dir = os.path.join(dataset_dir, 'label')
        self.image_paths = sorted(glob(os.path.join(images_dir, '*.nii.gz')))
        self.label_paths = sorted(glob(os.path.join(labels_dir, '*.nii.gz')))
        assert len(self.image_paths) == len(self.label_paths)

        subjects = []
        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                mri=ScalarImage(image_path),
                brain=LabelMap(label_path),
            )
            subjects.append(subject)
        self.dataset = tio.SubjectsDataset(subjects)

        print('Dataset size:', len(self.dataset), 'subjects')

        num_subjects = len(self.dataset)
        num_training_subjects = int(training_split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects

        num_split_subjects = num_training_subjects, num_validation_subjects
        self.training_subjects, self.validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

        self._transform()

    def _transform(self):
        self.landmarks = tio.HistogramStandardization.train(self.image_paths)

        self.training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad((48, 60, 48)),
            tio.RandomMotion(p=0.2),
            tio.HistogramStandardization({'mri': self.landmarks}),
            tio.RandomBiasField(p=0.3),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            tio.OneHot(),
        ])

        self.validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad((48, 60, 48)),
            tio.HistogramStandardization({'mri': self.landmarks}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.OneHot(),
        ])
        
    def get_landmarks(self):
        return self.landmarks

    def train_set(self):        
        training_set = tio.SubjectsDataset(
            self.training_subjects, transform=self.training_transform)

        print('Training set:', len(training_set), 'subjects')
        return training_set
    def val_set(self):
        validation_set = tio.SubjectsDataset(
            self.validation_subjects, transform=self.validation_transform)
        print('Validation set:', len(validation_set), 'subjects')
        return validation_set


def read_param(path):
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict
    

if __name__ == "__main__":
    # d = Datasets3D("/home/zhaozixiao/projects/datasets/flare21/FLARE21/imagesTr/", "/home/zhaozixiao/projects/datasets/flare21/FLARE21/labelsTr/")
    # print(len(d.train_set()))
    # print(len(d.val_set()))
    # print(os.getcwd())
    dl = Datasets3D('/home/zhaozixiao/projects/singa_local/singa-auto/test/dataset/ixi_tiny', 0.9)
    tr = dl.train_set()
    va = dl.val_set()