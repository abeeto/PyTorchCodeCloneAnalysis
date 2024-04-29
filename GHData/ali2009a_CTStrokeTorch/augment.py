import os
import glob
import shutil
import tempfile
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import gc
from monai.apps import download_and_extract
from monai.config import print_config
from monai.transforms import Affine, Rand3DElasticd, RandAffine, LoadNifti, Orientationd, Spacingd, LoadNiftid, AddChanneld

print_config()


class augmentor:

    def __init__(self, imageRepoPath, outputRepoPath):
        self.imageRepoPath = imageRepoPath
        self.outputRepoPath = outputRepoPath
        self.counter=0

    def fetchImagePaths(self):
        files = list()
        for subject_dir in glob.glob(os.path.join(self.imageRepoPath, "*")):
            subjectID = os.path.basename(subject_dir)
            imageFile = os.path.join(self.imageRepoPath, subjectID, "ct.nii.gz")		
            truthFile = os.path.join(self.imageRepoPath, subjectID, "truth.nii.gz")
            files.append({"image":imageFile, "label":truthFile})
        return files

    def agumentFiles(self, files):
        for pair in tqdm(files):
            self.augmentFile(pair)
    
    def augmentFile(self, pair):
        #self.scaleAugment(pair)
        #self.rotateAugment(pair)
        #self.translateAugment(pair)
        #self.shearAugment(pair)
        #self.randAffineAugment(pair)
        self.randElasticAugment(pair)

    def scaleAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        scale_params=[1.2,0.8]

        for p in scale_params:
            affine = Affine(scale_params=(p, p, p), padding_mode="reflection")
            print(p)
            new_img_matrix = affine(image_matrix, mode="nearest")
            new_lbl_matrix = affine(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 

    def rotateAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        d = np.pi/180
        rotate_params=[(0,0,15*d), (0,0,-15*d)]

        for p in rotate_params:
            affine = Affine(rotate_params=p, padding_mode="reflection")
            print(p)
            new_img_matrix = affine(image_matrix, mode="nearest")
            new_lbl_matrix = affine(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 



    def translateAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        d = np.pi/180
        translate_params=[(50,50,0), (-50,-50,0)]

        for p in translate_params:
            affine = Affine(translate_params=p, padding_mode="reflection")
            print(p)
            new_img_matrix = affine(image_matrix, mode="nearest")
            new_lbl_matrix = affine(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 


    def shearAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        d = np.pi/180
        shear_params=[(0.5, 0, 0, 0, 0, 0), (0, 0, 0.5, 0, 0, 0)]

        for p in shear_params:
            affine = Affine(shear_params=p, padding_mode="reflection")
            print(p)
            new_img_matrix = affine(image_matrix, mode="nearest")
            new_lbl_matrix = affine(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 


    def randAffineAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        d = np.pi/180
        affine_image = RandAffine(prob=1, rotate_range=(0,0,15*d), shear_range=(0.3, 0, 0.3, 0, 0, 0), translate_range=(50,50,2), scale_range=(0.1,0.1,0.1), padding_mode="border", mode="bilinear")
        affine_label = RandAffine(prob=1, rotate_range=(0,0,15*d), shear_range=(0.3, 0, 0.3, 0, 0, 0), translate_range=(50,50,2), scale_range=(0.1,0.1,0.1), padding_mode="border", mode="nearest")

        for i in range(5):
            print(i)
            affine.set_random_state(seed=i)
            new_img_matrix = affine_image(image_matrix, mode="nearest")
            affine.set_random_state(seed=i)
            new_lbl_matrix = affine_label(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 




    def randElasticAugment(self, pair):
        print("augmenting using elastic deformation")
        loader = LoadNiftid(keys=("image", "label"))
        data_dict = loader(pair)

        add_channel = AddChanneld(keys=["image", "label"])
        data_dict = add_channel(data_dict)
        
        #spacing = Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear", "nearest"))
        #data_dict = spacing(datac_dict)
        
        #orientation = Orientationd(keys=["image", "label"], axcodes="PLI")
        #data_dict = orientation(data_dict)
        
        d = np.pi/180

        rand_elastic = Rand3DElasticd(
            keys= ["image", "label"],
            mode=("nearest", "nearest"),
            prob=1.0,
            sigma_range=(8, 11),
            magnitude_range = (100, 200),
            rotate_range = (0,0,15*d),
            shear_range=(0.3, 0, 0.3, 0, 0, 0),
            translate_range=(50,50,2),
            scale_range=(0.1,0.1,0.1),
            padding_mode="border"
        )


        for i in range(10):
            deformed_data_dict = rand_elastic(data_dict)
            new_img_matrix, new_lbl_matrix = deformed_data_dict["image"][0], deformed_data_dict["label"][0]
            final_img = nib.Nifti1Image(new_img_matrix, data_dict["image_meta_dict"]["affine"])
            final_lbl = nib.Nifti1Image(new_lbl_matrix, data_dict["label_meta_dict"]["affine"])
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 
            gc.collect()


def main():
    aug = augmentor("data/original/train/","data/augmented/train/")
    files=  aug.fetchImagePaths()
    aug.agumentFiles(files)





