import glob
import os
import shutil
import nibabel as nib
import numpy as np

from monai.transforms import Resized
from monai.transforms import Affine, Rand3DElasticd, RandAffine, LoadNifti, Orientationd, Spacingd, LoadNiftid, AddChanneld, ScaleIntensityRanged, Resize
from tqdm import tqdm

DataPath = "/home/aliarab/scratch/CTStroke/segmentation/Data"
outPath  = "data/original"
#outPath  = "/local-scratch/aarab/CTStrokeTorch/data/original/train"

imageList = [ "ACT01" , "ACT02" , "ACT03" , "ACT04" , "ACT05" , "ACT06" , "ACT07" , "ACT08" , "ACT09" , "ACT10" , "ACT11" , "ACT13" , "ACT26" , "ACT28" , "ACT31" , "ACT12" , "ACT14" , "ACT15" , "ACT16" , "ACT166" , "ACT167" , "ACT168" , "ACT169" , "ACT17" , "ACT172" , "ACT173" , "ACT174" , "ACT175" , "ACT176" , "ACT18" , "ACT180" , "ACT181" , "ACT182" , "ACT183" , "ACT184" , "ACT185" , "ACT186" , "ACT187" , "ACT188" , "ACT189" , "ACT19" , "ACT190" , "ACT191" , "ACT192" , "ACT193" , "ACT194" , "ACT195" , "ACT197" , "ACT198" , "ACT20", "ACT66", "ACT77", "ACT107", "ACT114", "ACT118", "ACT125"]

needsFlip = ["ACT12" , "ACT14","ACT15","ACT17","ACT18", "ACT19", "ACT20", "ACT166", "ACT167", "ACT168", "ACT169", "ACT173", "ACT174", "ACT175", "ACT181", "ACT182", "ACT184", "ACT186", "ACT187", "ACT188", "ACT189", "ACT190", "ACT191", "ACT198"]


def main():
    for subject_folder in glob.glob(os.path.join(DataPath, "*")):
        
        truthPath= os.path.join(subject_folder, "groundTruth/Image.nii.gz")
        if not  os.path.exists(truthPath):
            continue
        subjectName = os.path.basename(os.path.normpath(subject_folder))
        if not subjectName in imageList:
        #if not subjectName in ["ACT15"]:
            continue
        print ("copying {}".format(subjectName))
        new_subject_folder = os.path.join(outPath,subjectName )
        if not os.path.exists(new_subject_folder):
            os.makedirs(new_subject_folder)
        truthPath= os.path.join(subject_folder, "groundTruth/Image.nii.gz")
        imagePath = os.path.join(subject_folder, "nii/Image.nii.gz")
        new_truthPath = os.path.join(new_subject_folder, "truth.nii.gz")
        new_imagePath  = os.path.join(new_subject_folder, "ct.nii.gz")
        fixFlip(imagePath, truthPath, new_imagePath, new_truthPath, subjectName)     

def fixFlip(in_img, in_label, out_img, out_label, subjectName):
    loader = LoadNifti()
    img, img_meta = loader(in_img)
    lbl, lbl_meta = loader(in_label)
    if subjectName in needsFlip:
        flipped_lbl= np.flipud(lbl)
    else:
        flipped_lbl= lbl
    flipped_lbl = nib.Nifti1Image(flipped_lbl, img_meta["affine"])
    intact_image = nib.Nifti1Image(img, img_meta["affine"])
    flipped_lbl.to_filename(out_label)
    intact_image.to_filename(out_img)

main()

